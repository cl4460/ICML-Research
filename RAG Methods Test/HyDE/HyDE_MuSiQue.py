import os
import time
import pandas as pd
import yaml
from tqdm import tqdm  
from openai import OpenAI
from HyDE import HyDEConfig, Promptor, HyDE


def get_gpt4_evaluation(ground_truth, model_response, openai_client):
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
        completion = openai_client.chat.completions.create(
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

def process_musique_questions(
    input_parquet: str,
    main_folder: str,
    output_parquet: str = None,
    openai_api_key: str = None
):
    
    openai_client = OpenAI(api_key=openai_api_key)
    df = pd.read_parquet(input_parquet)
    print(f"[INFO] Loaded MuSiQue data: {df.shape}")

    with open('Hconfig.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    base_config['config']['main_folder'] = main_folder

    hyde_config = HyDEConfig(base_config)
    promptor = Promptor(hyde_config)
    hyde_obj = HyDE(hyde_config, promptor)
    for col in ["retrieval_document", "response", "processing_time", "gpt4_evaluation", "binary_score"]:
        if col not in df.columns:
            df[col] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="MuSiQue-HyDE"):
        question = str(row['question'])
        start_time = time.time()
        retrieval_document = hyde_obj.e2e_search(question)

        end_time = time.time()
        processing_time = end_time - start_time
        if retrieval_document and isinstance(retrieval_document, list):
            best_hit = [retrieval_document[0]]
            response_data = hyde_obj.answer(best_hit, question)
        else:
            response_data = hyde_obj.answer([], question)

        df.at[idx, "retrieval_document"] = str(retrieval_document)
        df.at[idx, "response"] = response_data
        df.at[idx, "processing_time"] = processing_time

    # Use GPT-4 to compare the ground truth answer (df['answer']) with the HyDE response (df['response'])
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="MuSiQue-GPT4-Eval"):
        gpt_eval = get_gpt4_evaluation(
            ground_truth=str(row['answer']),
            model_response=str(row['response']),
            openai_client=openai_client
        )
        df.at[idx, "gpt4_evaluation"] = gpt_eval


    def to_binary_score(eval_str: str):
        if not eval_str:
            return 0
        eval_str_lc = eval_str.strip().lower()
        if eval_str_lc.startswith("yes"):
            return 1
        else:
            return 0

    df["binary_score"] = df["gpt4_evaluation"].apply(to_binary_score)
    total_count = len(df)
    num_correct = df["binary_score"].sum()
    accuracy = num_correct / total_count if total_count > 0 else 0.0
    avg_processing_time = df["processing_time"].mean() if total_count > 0 else 0.0

    print(f"\n[STATS] Accuracy: {accuracy:.2%} ({num_correct}/{total_count})")
    print(f"[STATS] Average processing time: {avg_processing_time:.2f} seconds")

    if output_parquet:
        df.to_parquet(output_parquet, index=False)
        print(f"[INFO] Results saved to: {output_parquet}")

    return df


if __name__ == "__main__":
    input_parquet_path = "/Users/chengze/Desktop/musique_ans_v1.0_dev_175sample.parquet"  
    main_folder_path = "/Users/chengze/Desktop/NaiveRAG_musique"             
    output_parquet_path = "/Users/chengze/Desktop/NaiveRAG_musique_response_results_175sample.parquet"


    df_result = process_musique_questions(
        input_parquet=input_parquet_path,
        main_folder=main_folder_path,
        output_parquet=output_parquet_path,
        openai_api_key= ""  
    )

    print("\n=== Final Data ===")
    print(df_result.head(10))
