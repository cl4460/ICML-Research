import os
import re
import json
from tqdm import tqdm
import pandas as pd

def calculate_metrics(pred_list, gold_list):
    """
    Calculate Precision, Recall, and F1 scores based on word intersection.
    """
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
    """
    Check if two strings share at least one common word.
    """
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words.intersection(b_words)) > 0


def extract_answer(input_string: str) -> str:
    """
    Extract the answer from the input string if it follows a specific pattern.

    Args:
        input_string (str): The string containing the answer.

    Returns:
        str: The extracted answer or the original string if no pattern is found.
    """
    match = re.search(r'The answer to the question is "(.*?)"', input_string)
    return match.group(1) if match else input_string


def convert_json_to_doc_data(json_file: str):
    """
    Read a JSON Lines file and convert it into a structured doc_data format.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries with keys 'query', 'model_answer', 'question_type', and 'answer'.
    """
    try:
        # 使用 Pandas 读取 JSON Lines 格式
        df = pd.read_json(json_file, lines=True)
        data = df.to_dict(orient='records')
        print("已成功以JSON Lines格式读取JSON文件。")
    except ValueError as e:
        raise ValueError(f"读取JSON文件失败: {e}")

    required_fields = ['question', 'answer', 'model_response']
    for field in required_fields:
        if not all(field in entry for entry in data):
            raise ValueError(f"JSON文件中的所有条目必须包含 '{field}' 字段。")

    # 如果 'question_type' 不存在，则设置为 'N/A'
    doc_data = []
    for entry in data:
        doc_data.append({
            "query": entry['question'],
            "model_answer": entry['model_response'],
            "question_type": entry.get('question_type', "N/A"),
            "answer": entry['answer']  # 添加 'answer' 字段
        })
    return doc_data


def read_ground_truth(doc_data: list):
    """
    Extract ground truth data from the doc_data list.

    Args:
        doc_data (list): List of dictionaries with 'query' and 'answer'.

    Returns:
        list: A list of dictionaries with keys 'query' and 'answer'.
    """
    ground_truth = []
    seen_queries = set()
    for entry in doc_data:
        query = entry['query']
        answer = entry['answer']
        if query not in seen_queries:
            ground_truth.append({"query": query, "answer": answer})
            seen_queries.add(query)
    return ground_truth


def evaluate_lightrag_multihop_responses(doc_data: list, ground_truth_data: list):
    """
    Evaluate the model responses against the ground truth using word intersection.

    Args:
        doc_data (list): List of dictionaries with 'query', 'model_answer', 'question_type', and 'answer'.
        ground_truth_data (list): List of dictionaries with 'query' and 'answer'.
    """
    type_data = {}
    overall_pred_list = []
    overall_gold_list = []
    gold_dict = {item['query']: item['answer'] for item in ground_truth_data}

    for entry in tqdm(doc_data, desc="Evaluating answers"):
        model_answer = entry['model_answer']
        if 'The answer' in model_answer:
            model_answer = extract_answer(model_answer)

        query = entry['query']
        gold = gold_dict.get(query, '')
        if gold:
            question_type = entry['question_type']
            if question_type not in type_data:
                type_data[question_type] = {'pred_list': [], 'gold_list': []}
            type_data[question_type]['pred_list'].append(model_answer)
            type_data[question_type]['gold_list'].append(gold)
            overall_pred_list.append(model_answer)
            overall_gold_list.append(gold)

    # Calculate and print metrics per question type
    for question_type, data in type_data.items():
        precision, recall, f1 = calculate_metrics(data['pred_list'], data['gold_list'])
        print(f"\nQuestion Type: {question_type}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1 Score:  {f1:.2f}")

    # Calculate and print overall metrics
    overall_precision, overall_recall, overall_f1 = calculate_metrics(
        overall_pred_list, overall_gold_list
    )
    print("\nOverall Metrics:")
    print(f" Precision: {overall_precision:.2f}")
    print(f" Recall:    {overall_recall:.2f}")
    print(f" F1 Score:  {overall_f1:.2f}")


def main():
    """
    Main function to execute the evaluation process:
      1. Read JSON file and convert to doc_data.
      2. Read ground truth data.
      3. Perform evaluation and print results.
    """
    # Specify the JSON file path
    json_path = os.path.expanduser("/Users/chengze/Desktop/filtered_Merged_HyDE_MultiHop.json")  # 修改为过滤后的文件

    # Check if the JSON file exists
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"无法找到JSON文件: {json_path}")

    # Convert JSON to doc_data
    print("Converting JSON to doc_data...")
    doc_data = convert_json_to_doc_data(json_path)

    # Read ground truth data
    print("Reading ground truth data...")
    ground_truth_data = read_ground_truth(doc_data)

    # Start evaluation
    print("Starting evaluation...")
    evaluate_lightrag_multihop_responses(doc_data, ground_truth_data)


if __name__ == "__main__":
    main()
