import os
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
import yaml
import tiktoken
from datetime import datetime
from TGRAG import TGConfig,TGSearch

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

OPENAI_API_KEY = "REMOVED"  
HOTPOT_FOLDER = "/Users/chengze/Desktop/TGRAG_MultiHop_indexing"
INPUT_PARQUET = "/Users/chengze/Desktop/MultiHopRAG_375_sampled.parquet"
INPUT_DIR = os.path.join(HOTPOT_FOLDER, "output")


def process_question_batch(df, search_engine):
    retrieval_tokens_list = []
    model_response_list = []
    processing_time_list = []
    retrieval_docs_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        question = row['query']
        start_time = time.time()
        
        # 检索并生成回答
        response,retrieved_info = search_engine.answer(question,id_type=True,retrievel=True)
        processing_time = time.time() - start_time
        

        model_response_list.append(response)
        retrieval_tokens_list.append(count_tokens(retrieved_info))
        processing_time_list.append(processing_time)
        retrieval_docs_list.append(retrieved_info)
            
            

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
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    TGRAG_config = TGConfig(config_data)
    tgrag_search = TGSearch(TGRAG_config)

    print("\nStarting batch processing...")
    df = process_question_batch(df, tgrag_search)
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
