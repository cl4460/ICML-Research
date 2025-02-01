import os
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
from datetime import datetime
from Graphrag.Graghrag import init_global_search, init_local_search,count_tokens


OPENAI_API_KEY = "REMOVED"  
HOTPOT_FOLDER = "/Users/chengze/Desktop/GraphRAG_MultiHopRAG"
INPUT_PARQUET = "/Users/chengze/Desktop/MultiHopRAG_375_sampled.parquet"
INPUT_DIR = os.path.join(HOTPOT_FOLDER, "output")


def process_question_batch(df, search_engine, openai_client, output_file):
    retrieval_tokens_list = []
    model_response_list = []
    processing_time_list = []
    retrieval_docs_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['query']
        start_time = time.time()
        result = search_engine.search(question)
        processing_time = time.time() - start_time
        model_response = result.response
        retrieval_docs = result.context_text

        model_response_list.append(model_response)
        retrieval_tokens_list.append(count_tokens(retrieval_docs))
        processing_time_list.append(processing_time)
        retrieval_docs_list.append(retrieval_docs)
            
            
    df['model_response'] = model_response_list
    df['retrieval_tokens'] = retrieval_tokens_list
    df['processing_time'] = processing_time_list
    df['retrieval_docs'] = retrieval_docs_list
    return df

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

    search_engine = init_local_search(INPUT_DIR,api_key=OPENAI_API_KEY)
    print("\nStarting batch processing...")
    df = process_question_batch(df, search_engine, OpenAI(api_key=OPENAI_API_KEY), output_file)
    total_questions = len(df)
    avg_time = sum(df['processing_time']) / total_questions if total_questions > 0 else 0
    avg_tokens = sum(df['retrieval_tokens']) / total_questions if total_questions > 0 else 0
    print(f"\nProcessing completed!")
    print(f"Total questions processed: {total_questions}")
    print(f"Average processing time per question: {avg_time:.2f} seconds")
    print(f"Average retrieval tokens per question: {avg_tokens:.2f}")
    print(f"Results saved to: {output_file}")

    # 分析结果
    df.to_parquet(os.path.join(INPUT_DIR, "results", f"qa_results_{timestamp}.parquet"))

if __name__ == "__main__":
    main()
