import os
import time
import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm
import json
from datetime import datetime
from dotenv import load_dotenv  # 如果使用 .env 文件管理环境变量
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_covariates
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

OPENAI_API_KEY = ""  
MuSiQue_FOLDER = "/Users/chengze/Desktop/GraphRAG_MuSiQue"
INPUT_PARQUET = "/Users/chengze/Desktop/musique_ans_v1.0_dev_175sample.parquet"
OUTPUT_DIR = os.path.join(MuSiQue_FOLDER, "output")
LANCEDB_URI = os.path.join(OUTPUT_DIR, "lancedb")

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

def process_question_batch(df, search_engine, openai_client, output_file):
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing questions"):
        try:
            question = row['question']
            ground_truth = row['answer']
            start_time = time.time()
            
            # retrival and generate answer
            result = search_engine.search(question)
            model_response = result.response
            retrieval_docs = result.context_data.get("entities", [])

            # call gpt-4 to evaluate
            evaluation = get_gpt4_evaluation(openai_client, ground_truth, model_response)
            processing_time = time.time() - start_time
            # format retrieval docs
            docs = ''
            if isinstance(retrieval_docs, list):
                for doc in retrieval_docs:
                    docs += str(doc) + ' '
            else:
                docs = str(retrieval_docs)
            
            result_dict = {
                'question_id': idx,
                'question': question,
                'ground_truth': ground_truth,
                'model_response': model_response,
                'retrieval_docs': docs.strip(),
                'evaluation': evaluation,
                'processing_time': processing_time
            }
            results.append(result_dict)
            
            # 实时保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            # 添加延时以避免 API 限制
            time.sleep(1)
        except Exception as e:
            print(f"\nError processing question {idx}: {str(e)}")
            continue
    
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
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建带时间戳的输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"qa_results_{timestamp}.json")

    # 加载必要的索引文件
    final_nodes_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "create_final_nodes.parquet"))
    final_entities_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "create_final_entities.parquet"))
    relationship_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "create_final_relationships.parquet"))
    report_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "create_final_community_reports.parquet"))
    text_unit_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "create_final_text_units.parquet"))

    # 如果存在 covariates，则加载
    covariates_df = None
    covariates = None
    covariates_path = os.path.join(OUTPUT_DIR, "create_final_covariates.parquet")
    if os.path.exists(covariates_path):
        covariates_df = pd.read_parquet(covariates_path)
        covariates = read_indexer_covariates(covariates_df)
        print(f"Covariates loaded: {len(covariates_df)}")
    else:
        print("No covariates found. Setting covariates to None.")

    # 读取实体、关系、报告和文本单元
    community_level = 2  # 或根据需要设为 None
    entities = read_indexer_entities(final_nodes_df, final_entities_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    reports = read_indexer_reports(report_df, final_nodes_df, community_level)
    text_units = read_indexer_text_units(text_unit_df)

    # 初始化 LanceDB 向量数据库
    description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    # 初始化 LLM 和 Embedding 模型
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )
    
    token_encoder = tiktoken.get_encoding("cl100k_base")
    
    text_embedder = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    # 创建本地搜索上下文构建器
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,  # 如果没有 covariates，就设为 None
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # 如果 vector store 使用 entity 标题作为主键，可改为 EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    # 定义上下文参数（与官方指南保持一致）
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 6000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    # 定义 LLM 参数（与官方指南保持一致）
    llm_params = {
        "max_tokens": 2_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.0,
    }

    # 初始化本地搜索引擎（与官方指南保持一致）
    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    # 处理问题并保存结果
    print("\nStarting batch processing...")
    results = process_question_batch(df, search_engine, OpenAI(api_key=OPENAI_API_KEY), output_file)
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
