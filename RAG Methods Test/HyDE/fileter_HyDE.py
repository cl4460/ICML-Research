import json

# 文件路径
input_file_path = '/Users/chengze/Desktop/Merged_HyDE_MultiHop_with_gt.json'
output_file_path = '/Users/chengze/Desktop/filtered_Merged_HyDE_MultiHop.json'

# 初始化一个列表来存储过滤后的数据
filtered_data = []

# 逐行读取 JSON Lines 文件并进行过滤
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line_number, line in enumerate(infile, start=1):
        try:
            entry = json.loads(line)
            if entry.get('question_type') != 'comparison_query':
                filtered_data.append(entry)
        except json.JSONDecodeError as e:
            print(f"在第 {line_number} 行遇到 JSON 解码错误: {e}")
            continue

# 将过滤后的数据写入新的 JSON Lines 文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for entry in filtered_data:
        json_line = json.dumps(entry, ensure_ascii=False)
        outfile.write(json_line + '\n')

print(f"过滤后的文件已保存到 {output_file_path}")
