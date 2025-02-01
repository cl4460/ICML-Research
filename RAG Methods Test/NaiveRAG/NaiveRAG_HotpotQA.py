import os
import time
import json
import yaml
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from NaiveRAG.build.Naive_config import NaiveConfig
from NaiveRAG.search.Naive_search import Naive_search
from openai import OpenAI 
import tiktoken

########################################
# 1) 简单的 token 计数函数
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
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text)
    return len(tokens)

########################################
# 2) GPT-4 评价函数
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
# 3) 批量处理函数：只做 query → response
########################################
def process_question_batch(df, naive_search, openai_client):
    """
    处理问题批次，使用 NaiveRAG 进行检索与回答，并使用 GPT-4 进行评价。
    返回结果列表。
    """
    results = []  # 用于存储每条记录的输出
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        try:
            question = row['question']
            ground_truth = row['answer']
            start_time = time.time()
            
            # 1) 检索与生成回答
            model_response, retrieved_info = naive_search.answer(question, retrievel=True)
            processing_time = time.time() - start_time
            
            # 2) GPT-4 评价
            evaluation = get_gpt4_evaluation(openai_client, ground_truth, model_response)

            # 3) 确保 'retrieved_info' 是字符串
            if isinstance(retrieved_info, list):
                # 将list中的所有文档用换行拼接
                retrieved_info_str = "\n".join([str(doc) for doc in retrieved_info])
            elif retrieved_info is None:
                # 如果是None，赋值为空字符串
                retrieved_info_str = ""
            else:
                # 如果是其他类型，直接转成字符串
                retrieved_info_str = str(retrieved_info)

            # 4) 计算 tokens
            retrieval_tokens = count_tokens(retrieved_info_str)

            # 5) 构造本条结果的字典
            result_dict = {
                'question_id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'model_response': model_response,
                'retrieved_list': retrieved_info,
                'evaluation': evaluation,
                'processing_time': processing_time,
                'retrieval_tokens': retrieval_tokens
            }
            results.append(result_dict)
            
        except Exception as e:
            print(f"\nError processing question {idx}: {str(e)}")
            continue

    return results

########################################
# 4) 结果分析函数
########################################
def analyze_results(json_file_path: str):
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
    avg_tokens = sum(r['retrieval_tokens'] for r in results) / total_questions

    print(f"\n[RESULTS ANALYSIS]")
    print(f"Total questions analyzed: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Incorrect answers: {incorrect_answers}")
    print(f"Unclear answers: {unclear_answers}")
    print(f"Accuracy rate: {(correct_answers / total_questions) * 100:.2f}%")
    print(f"Average processing time per question: {avg_time:.2f} seconds")
    print(f"Average retrieval tokens per question: {avg_tokens:.2f}")

########################################
# 5) 主函数
########################################
def main():
    # ========== 配置信息 ==========
    OPENAI_API_KEY = "REMOVED"
    
    # 读取配置文件
    with open('Nconfig.yaml', 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    # 初始化 NaiveRAG 配置与搜索对象
    Naive_config = NaiveConfig(config_data)
    naive_search = Naive_search(Naive_config)

    # 设置文件路径
    HOTPOT_FOLDER = "/Users/chengze/Desktop/NaiveRAG_MuSiQue_indexing"
    INPUT_PARQUET = "/Users/chengze/Desktop/musique_ans_v1.0_dev_175sample.parquet"
    
    # 读取 DataFrame
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Total questions to process: {len(df)}")

    # 创建输出目录
    OUTPUT_DIR = os.path.join(HOTPOT_FOLDER, "output")
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成输出 JSON 文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"qa_results_{timestamp}.json")
    
    # 初始化 OpenAI 客户端
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # ========== 执行检索+回答 ==========
    print("\nStarting batch processing with NaiveRAG and GPT-4 evaluation...")
    results = process_question_batch(df, naive_search, openai_client)
    
    # ========== 写出 JSON ==========
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to JSON: {output_file}")
    
    # ========== 结果分析 ==========
    analyze_results(json_file_path=output_file)

if __name__ == "__main__":
    main()
