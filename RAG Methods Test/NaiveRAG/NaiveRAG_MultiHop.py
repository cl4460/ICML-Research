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


def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text)
    return len(tokens)


def process_question_batch(df, naive_search, openai_client):
    results = [] 
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        try:
            question = row['query']
            ground_truth = row['answer']
            start_time = time.time()
            model_response, retrieved_info = naive_search.answer(question, retrievel=True)
            processing_time = time.time() - start_time
            question_type = row['question_type']

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
                'processing_time': processing_time,
                'retrieval_tokens': retrieval_tokens,
                'question_type' : question_type
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


    avg_time = sum(r['processing_time'] for r in results) / total_questions
    avg_tokens = sum(r['retrieval_tokens'] for r in results) / total_questions

    print(f"\n[RESULTS ANALYSIS]")
    print(f"Total questions analyzed: {total_questions}")
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
    HOTPOT_FOLDER = "/Users/chengze/Desktop/NaiveRAG_MultiHop_indexing"
    INPUT_PARQUET = "/Users/chengze/Desktop/MultiHopRAG_375_sampled.parquet"
    
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
