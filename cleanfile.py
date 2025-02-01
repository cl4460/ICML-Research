#!/usr/bin/env python3
import os

# 定义需要删除敏感信息的关键词（区分大小写）

def clean_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件 {filepath} 失败：{e}")
        return

    # 仅保留不包含敏感关键词的行
    new_lines = []
    removed = False
    for line in lines:
        if any(keyword in line for keyword in sensitive_keywords):
            removed = True
            # 如果你希望保留行的其他部分，也可以在此处做更细粒度的替换
            continue  # 此处直接跳过该行
        new_lines.append(line)

    if removed:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"已清理文件：{filepath}")
        except Exception as e:
            print(f"写入文件 {filepath} 失败：{e}")

def main():
    # 遍历当前目录及所有子目录
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith((".py", ".ipynb", ".yaml")):
                fullpath = os.path.join(root, file)
                clean_file(fullpath)

if __name__ == '__main__':
    main()
