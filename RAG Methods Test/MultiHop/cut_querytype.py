# import pandas as pd

# # 读取文件的路径
# file1_path = '/Users/chengze/Desktop/TGRAG_multihop_175sample_responses_evaluated2.parquet'
# file2_path = '/Users/chengze/Desktop/LightRAG_MultiHopRAG_175sample_responses.parquet'

# # 加载 parquet 文件
# df1 = pd.read_parquet(file1_path)
# df2 = pd.read_parquet(file2_path)

# # 删除 question_type 为 'comparison_query' 的行
# df1_filtered = df1[df1['question_type'] != 'comparison_query']
# df2_filtered = df2[df2['question_type'] != 'comparison_query']

# # 将过滤后的数据分别保存为新文件
# df1_filtered.to_parquet('/Users/chengze/Desktop/filtered_TGRAG_multihop.parquet', index=False)
# df2_filtered.to_parquet('/Users/chengze/Desktop/filtered_LightRAG_MultiHopRAG.parquet', index=False)
import json

# 文件路径
input_file_path = '/Users/chengze/Desktop/NaiveRAG.json'
output_file_path = '/Users/chengze/Desktop/filtered_NaiveRAG.json'

# 加载 JSON 文件
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 删除 question_type 为 'comparison_query' 的记录
filtered_data = [entry for entry in data if entry.get('question_type') != 'comparison_query']

# 保存处理后的数据到新的 JSON 文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

print(f"过滤后的文件已保存到 {output_file_path}")
