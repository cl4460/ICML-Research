import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def merge_question_type(json_file, parquet_file, output_file):
    """
    Merge the 'question_type' from the Parquet file into the JSON file based on 'question' and 'query'.
    
    Args:
        json_file (str): Path to the JSON file containing model responses.
        parquet_file (str): Path to the Parquet file containing ground truth data.
        output_file (str): Path where the merged JSON file will be saved.
    """
    # Read the Parquet file
    print("Reading Parquet file...")
    try:
        df_parquet = pd.read_parquet(parquet_file)
    except Exception as e:
        raise ValueError(f"Error reading Parquet file: {e}")
    
    # Print columns to debug
    print("Parquet file columns:", df_parquet.columns.tolist())
    
    # Ensure 'query' and 'question_type' exist
    if 'query' not in df_parquet.columns:
        raise ValueError("Parquet file does not contain 'query' column.")
    if 'question_type' not in df_parquet.columns:
        raise ValueError("Parquet file does not contain 'question_type' column.")
    
    # Select 'query' and 'question_type' for merging
    df_parquet = df_parquet[['query', 'question_type']]
    
    # Read the JSON file
    print("Reading JSON file...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error reading JSON file: {e}")
    
    df_json = pd.DataFrame(data_json)
    
    # Print JSON columns to debug
    print("JSON file columns:", df_json.columns.tolist())
    
    # Ensure 'question' exists in JSON
    if 'question' not in df_json.columns:
        raise ValueError("JSON file does not contain 'question' field.")
    
    # Print first few JSON entries to verify
    print("JSON DataFrame head:")
    print(df_json.head())
    
    # Standardize 'question' and 'query' for matching
    df_json['question'] = df_json['question'].astype(str).str.strip().str.lower()
    df_parquet['query'] = df_parquet['query'].astype(str).str.strip().str.lower()
    
    # Check how many 'question' in JSON match 'query' in Parquet
    matched_questions = df_json['question'].isin(df_parquet['query']).sum()
    print(f"Number of matched questions: {matched_questions} out of {len(df_json)}")
    
    if matched_questions == 0:
        print("No matching questions found between JSON and Parquet files. Please check the 'question' and 'query' fields for consistency.")
    
    # Merge the two DataFrames on 'question' (JSON) and 'query' (Parquet)
    print("Merging data...")
    df_merged = pd.merge(df_json, df_parquet, left_on='question', right_on='query', how='left')
    
    # Fill missing 'question_type' with 'N/A'
    df_merged['question_type'] = df_merged['question_type'].fillna('N/A')
    
    # Print merged DataFrame columns and some stats
    print("Merged DataFrame columns:", df_merged.columns.tolist())
    num_question_type = df_merged['question_type'].notna().sum()
    num_na = df_merged['question_type'].isna().sum()
    print(f"Number of questions with 'question_type': {num_question_type}")
    print(f"Number of questions with 'question_type' as 'N/A': {num_na}")
    
    # Optionally, drop the 'query' column from the merged DataFrame
    df_merged = df_merged.drop(columns=['query'])
    
    # Save the merged DataFrame to a new JSON file using the standard json module
    print(f"Saving merged data to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(df_merged.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving merged JSON file: {e}")
    
    print("Merging completed successfully.")

def main():
    """
    Main function to execute the merging process.
    """
    # Specify the file paths
    json_file = os.path.expanduser("/Users/chengze/Desktop/HyDE_MultiHop.json")  # Update as needed
    parquet_file = os.path.expanduser("/Users/chengze/Desktop/MultiHopRAG_375_sampled.parquet")  # Update as needed
    output_file = os.path.expanduser("/Users/chengze/Desktop/query_type_HyDE_MultiHop.json")  # Update as needed
    
    # Check if input files exist
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"Cannot find the JSON file: {json_file}")
    if not os.path.isfile(parquet_file):
        raise FileNotFoundError(f"Cannot find the Parquet file: {parquet_file}")
    
    # Perform the merge
    merge_question_type(json_file, parquet_file, output_file)

if __name__ == "__main__":
    main()
