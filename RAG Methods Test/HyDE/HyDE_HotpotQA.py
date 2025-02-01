import os
import time
import json
import tiktoken  # 用于 count_tokens
import yaml
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

# ========== 引入 HyDE ==========
from HyDE import HyDE, HyDEConfig, Promptor


########################################
# 简单的 token 计数函数 (可选)
########################################
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string using tiktoken
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for tokenization (default: "gpt-4")
        
    Returns:
        int: Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 如果找不到对应的 model，就用 "cl100k_base"
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


########################################
# GPT-4 评价函数 (保持不变)
########################################
def get_gpt4_evaluation(client: OpenAI, ground_truth: str, model_response: str) -> str:
    system_prompt = (
        "You are a helpful assistant. Please evaluate if the response matches the reference answer."
    )
    user_prompt = f"""Instructions
You will receive a ground truth answer (referred to as Answer) and a model-generated answer (referred to as Response).
Your task is to compare the two and determine whether they align.

Note: The ground truth answer may sometimes be embedded within the model-generated answer.
You need to carefully analyze and discern whether they align.

Your Output:
If the two answers align, respond with yes.
If they do not align, respond with no.
If you are very uncertain, respond with unclear.

Your response should first include yes, no, or unclear, followed by an explanation.

Example 1
Answer: Houston Rockets
Response: The basketball player who was drafted 18th overall in 2001 is Jason Collins, who was selected by the Houston Rockets.
Expected output: yes

Example 2
Answer: no
Response: Yes, both Variety and The Advocate are LGBT-interest magazines.
          The Advocate is explicitly identified as an American LGBT-interest magazine,
          while Variety, although primarily known for its coverage of the entertainment industry,
          also addresses topics relevant to the LGBT community.
Expected output: no

Input Data Format
Ground Truth Answer: {ground_truth}
Model Generated Answer: {model_response}

Expected Output
yes, no, or unclear
An explanation of your choice.

Output:
"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        evaluation_text = completion.choices[0].message.content.strip()
        return evaluation_text
    except Exception as e:
        print(f"Error calling GPT-4: {e}")
        return f"ERROR: {str(e)}"


########################################
# 使用 HyDE 方法
########################################
def process_question_batch(df, hyde_obj, openai_client, output_file):
    """
    处理问题批次，使用 HyDE 检索并生成回答，然后调用 GPT-4 进行评价，将结果写入 JSON。
    不再依赖 GraphRAG 的 init_local_search 或 context_text, .response 等属性。
    """
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        try:
            question = row['question']
            ground_truth = row['answer']  # 如果 DataFrame 中存储了正确答案
            start_time = time.time()
            
            ######################
            # 1) 检索 (HyDE.e2e_search)
            ######################
            retrieval_docs = hyde_obj.e2e_search(question)
            # retrieval_docs 应该是一个 list，如果没有检索到可能是空列表

            # 记录检索时间
            processing_time = time.time() - start_time

            ######################
            # 2) 生成回答 (HyDE.answer)
            ######################
            if retrieval_docs and isinstance(retrieval_docs, list):
                # 取第一个检索结果
                best_hit = [retrieval_docs[0]]
                model_response = hyde_obj.answer(best_hit, question)
            else:
                # 若无检索结果，就传空列表
                model_response = hyde_obj.answer([], question)

            # 3) 调用 GPT-4 进行评价
            evaluation = get_gpt4_evaluation(openai_client, ground_truth, model_response)

            # 构造结果字典
            # 为了便于查看，也可把 retrieval_docs 改为 str(retrieval_docs)
            retrieval_docs_str = str(retrieval_docs) if isinstance(retrieval_docs, list) else ""
            result_dict = {
                'question_id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'model_response': model_response,
                'retrieval_docs': retrieval_docs_str,
                'evaluation': evaluation,
                'processing_time': processing_time,
                'retrieval_tokens': count_tokens(retrieval_docs_str)
            }

            results.append(result_dict)

            # 实时保存结果到 JSON 文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 可根据需要添加延时以避免 API 限制
            time.sleep(1)

        except Exception as e:
            print(f"\nError processing question {idx}: {str(e)}")
            continue
    
    return results


########################################
# 分析结果 (accuracy, time)
########################################
def analyze_results(json_file_path: str):
    """
    分析 JSON 结果文件，计算准确率、平均处理时间等统计信息。
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        return

    total_questions = len(results)
    if total_questions == 0:
        print("No results to analyze.")
        return

    correct_answers = sum(1 for r in results if r['evaluation'].lower().startswith('yes'))
    incorrect_answers = sum(1 for r in results if r['evaluation'].lower().startswith('no'))
    unclear_answers = sum(1 for r in results if r['evaluation'].lower().startswith('unclear'))
    avg_time = sum(r['processing_time'] for r in results) / total_questions

    print(f"\n[RESULTS ANALYSIS]")
    print(f"Total questions analyzed: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Incorrect answers: {incorrect_answers}")
    print(f"Unclear answers: {unclear_answers}")
    print(f"Accuracy rate: {(correct_answers / total_questions) * 100:.2f}%")
    print(f"Average processing time per question: {avg_time:.2f} seconds")


def main():
    """
    入口点
    """
    # ========= 配置路径 (根据需要自行修改) =========
    input_parquet_path = "/Users/chengze/Desktop/musique_ans_v1.0_dev_175sample.parquet"
    main_folder_path = "/Users/chengze/Desktop/NaiveRAG_MuSiQue"

    # 读取 Parquet
    df = pd.read_parquet(input_parquet_path)
    print(f"Total questions to process: {len(df)}")

    # 输出目录
    output_dir = os.path.join(main_folder_path, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出 JSON 文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"qa_results_{timestamp}.json")

    # ========== 初始化 HyDE 对象 ==========
    # 需要有 Hconfig.yaml
    import yaml
    with open('Hconfig.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    base_config['config']['main_folder'] = main_folder_path
    
    hyde_config = HyDEConfig(base_config)
    promptor = Promptor(hyde_config)
    hyde_obj = HyDE(hyde_config, promptor)

    # 初始化 GPT-4 客户端

    # 处理批量问题
    print("\nStarting batch processing with HyDE + GPT-4 evaluation...")
    results = process_question_batch(df, hyde_obj, openai_client, output_file)

    # 打印简单统计
    total_questions = len(results)
    if total_questions > 0:
        avg_time = sum(r['processing_time'] for r in results) / total_questions
    else:
        avg_time = 0

    print(f"\nProcessing completed!")
    print(f"Total questions processed: {total_questions}")
    print(f"Average processing time per question: {avg_time:.2f} seconds")
    print(f"Results saved to: {output_file}")

    # 分析结果
    analyze_results(json_file_path=output_file)


if __name__ == "__main__":
    main()
