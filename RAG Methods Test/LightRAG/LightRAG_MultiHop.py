import os
import re
import pandas as pd
from tqdm import tqdm


def calculate_metrics(pred_list, gold_list):
    tp = sum(1 for pred, gold in zip(pred_list, gold_list) 
             if has_intersection(pred.lower(), gold.lower()))
    fp = sum(1 for pred, gold in zip(pred_list, gold_list) 
             if not has_intersection(pred.lower(), gold.lower()))
    fn = len(gold_list) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def has_intersection(a: str, b: str) -> bool:
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words.intersection(b_words)) > 0

def extract_answer(input_string: str) -> str:
    match = re.search(r'The answer to the question is "(.*?)"', input_string)
    return match.group(1) if match else input_string

def convert_parquet_to_doc_data(parquet_file: str):
    df = pd.read_parquet(parquet_file)
    required_cols = ['query', 'answer', 'model_response']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Parquet 文件必须包含 '{col}' 列。")
    if 'question_type' not in df.columns:
        df['question_type'] = "N/A"
    doc_data = []
    for _, row in df.iterrows():
        doc_data.append({
            "query": row['query'],
            "model_answer": row['model_response'],
            "question_type": row['question_type']
        })
    return doc_data

def read_ground_truth(parquet_file: str):
    df = pd.read_parquet(parquet_file)
    if 'query' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Parquet 文件必须包含 'query' 和 'answer' 列。")
    
    ground_truth = df[['query', 'answer']].drop_duplicates().to_dict(orient='records')
    return ground_truth

def evaluate_lightrag_multihop_responses(doc_data: list, query_data: list):
    type_data = {}
    overall_pred_list = []
    overall_gold_list = []
    gold_dict = {item['query']: item['answer'] for item in query_data}
    for d in tqdm(doc_data, desc="Evaluating answers"):
        model_answer = d['model_answer']
        if 'The answer' in model_answer:
            model_answer = extract_answer(model_answer)
        
        query = d['query']
        gold = gold_dict.get(query, '')
        if gold:
            question_type = d['question_type']
            if question_type not in type_data:
                type_data[question_type] = {'pred_list': [], 'gold_list': []}
            type_data[question_type]['pred_list'].append(model_answer)
            type_data[question_type]['gold_list'].append(gold)
            overall_pred_list.append(model_answer)
            overall_gold_list.append(gold)
    
    for question_type, data in type_data.items():
        precision, recall, f1 = calculate_metrics(data['pred_list'], data['gold_list'])
        print(f"\nQuestion Type: {question_type}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1 Score:  {f1:.2f}")
    
    overall_precision, overall_recall, overall_f1 = calculate_metrics(
        overall_pred_list, overall_gold_list
    )
    print("\nOverall Metrics:")
    print(f" Precision: {overall_precision:.2f}")
    print(f" Recall:    {overall_recall:.2f}")
    print(f" F1 Score:  {overall_f1:.2f}")

def main():
    parquet_path = os.path.expanduser("/Users/chengze/Desktop/GraphRAG_MultiHop.parquet")
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Cannot find the Parquet File: {parquet_path}")

    print("Converting Parquet to doc_data...")
    doc_data = convert_parquet_to_doc_data(parquet_path)
    print("Reading ground truth data...")
    query_data = read_ground_truth(parquet_path)
    print("Starting evaluation...")
    evaluate_lightrag_multihop_responses(doc_data, query_data)

if __name__ == "__main__":
    main()
