import os
import time
import yaml
import pandas as pd
from tqdm import tqdm

# ======== TGRAG 相关 ========
from TGRAG import TGSearch, TGConfig

# ======== 1) 读取 TGRAG 配置 ========
config_path = "/Users/chengze/PycharmProjects/PythonProject/TG_rag_develop/config.yaml"
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

TGRAG_config = TGConfig(config)
TGRAG_search = TGSearch(TGRAG_config)

# ======== 2) 加载 Parquet 文件 ========
file_path = "/Users/chengze/Desktop/175_annotations_writing_with_citation.parquet"

# 确保输入文件存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found: {file_path}")

# 读取数据
df = pd.read_parquet(file_path)
test_df = df.copy()  # 处理所有175条数据

# ======== 3) 批量处理问题 ========
retrieved_contexts = []
responses = []
processing_time = []

# 遍历测试数据
for index in tqdm(range(len(test_df)), desc="Processing samples"):
    try:
        # 从 DataFrame 中读取问题
        question = test_df.iloc[index]['question']
        
        start_time = time.time()
        # 使用 TGRAG 搜索来获取回答
        response, retrieved_context = TGRAG_search.answer(question, retrievel=True)
        elapsed_time = time.time() - start_time

        # 记录检索结果
        retrieved_contexts.append(retrieved_context)
        responses.append(response)
        processing_time.append(elapsed_time)
        
    except Exception as e:
        print(f"Error processing question at index {index}: {str(e)}")
        retrieved_contexts.append(None)
        responses.append(None)
        processing_time.append(None)

# ======== 4) 写回结果并保存 ========
test_df['retrieved_context'] = retrieved_contexts
test_df['response'] = responses
test_df['processing_time'] = processing_time

# 确保输出目录存在
output_dir = os.path.dirname("/Users/chengze/Desktop")
os.makedirs(output_dir, exist_ok=True)

# 保存结果
output_file_path = "/Users/chengze/Desktop/TGRAG_writing_query-response.parquet"
test_df.to_parquet(output_file_path, index=False)

# ======== 5) 输出统计信息 ========
# 计算并打印平均处理时长（排除None值）
valid_times = [t for t in processing_time if t is not None]
if valid_times:
    average_processing_time = sum(valid_times) / len(valid_times)
    print(f"Average processing time: {average_processing_time:.2f} seconds")
else:
    print("No valid processing times recorded")

print(f"[INFO] Results saved to {output_file_path}")

# 显示处理结果摘要
print("\nProcessing Summary:")
print(f"Total samples processed: {len(test_df)}")
print(f"Successful responses: {len([r for r in responses if r is not None])}")
print(f"Failed responses: {len([r for r in responses if r is None])}")
