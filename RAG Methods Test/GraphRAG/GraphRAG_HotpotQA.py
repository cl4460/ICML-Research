import os
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
from datetime import datetime
from dotenv import load_dotenv  # 如果使用 .env 文件管理环境变量
from Graphrag.Graghrag import init_global_search,count_tokens
import asyncio


HOTPOT_FOLDER = "/Users/chengze/Desktop/GraphRAG_Musique"
INPUT_PARQUET = "/Users/chengze/Desktop/musique_ans_v1.0_dev_175sample.parquet"
INPUT_DIR = os.path.join(HOTPOT_FOLDER, "output")



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

async def process_question_batch(df, search_engine, openai_client, output_file):
    """
    处理问题批次，进行检索、生成回答、调用 GPT-4 评价，并保存结果。
    """
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['question']
        ground_truth = row['answer']
        start_time = time.time()
        
        # 检索并生成回答
        result = await search_engine.asearch(question)
        processing_time = time.time() - start_time
        model_response = result.response
        
        # 调用 GPT-4 进行评价
        evaluation = get_gpt4_evaluation(openai_client, ground_truth, model_response)
        result_dict = {
            'question_id': idx,
            'question': question,
            'ground_truth': ground_truth,
            'model_response': model_response,
            'retrieval_docs': result.context_text,
            'evaluation': evaluation,
            'processing_time': processing_time,
            'retrieval_tokens': result.prompt_tokens
        }
        results.append(result_dict)
        
        # 实时保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # 添加延时以避免 API 限制
        time.sleep(1)
    # except Exception as e:
    #     print(f"\nError processing question {idx}: {str(e)}")
    #     continue

    return results

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
    correct_answers = sum(1 for r in results if r['evaluation'].lower().startswith('yes'))
    incorrect_answers = sum(1 for r in results if r['evaluation'].lower().startswith('no'))
    unclear_answers = sum(1 for r in results if r['evaluation'].lower().startswith('unclear'))
    avg_time = sum(r['processing_time'] for r in results) / total_questions if total_questions > 0 else 0

    print(f"\n[RESULTS ANALYSIS]")
    print(f"Total questions analyzed: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Incorrect answers: {incorrect_answers}")
    print(f"Unclear answers: {unclear_answers}")
    print(f"Accuracy rate: {(correct_answers/total_questions)*100:.2f}%")
    print(f"Average processing time per question: {avg_time:.2f} seconds")

def main():
    # 读取 HotpotQA 数据集
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Total questions to process: {len(df)}")
    
    # 创建输出目录（如果不存在）
    results_dir = os.path.join(INPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建带时间戳的输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"qa_results_{timestamp}.json")

    print("\nStarting batch processing...")
    total_questions = len(results)
    avg_time = sum(r['processing_time'] for r in results) / total_questions if total_questions > 0 else 0
    
    print(f"\nProcessing completed!")
    print(f"Total questions processed: {total_questions}")
    print(f"Average processing time per question: {avg_time:.2f} seconds")
    print(f"Results saved to: {output_file}")

    # 分析结果
    analyze_results(json_file_path=output_file)

if __name__ == "__main__":
    main()
