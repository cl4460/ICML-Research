import pandas as pd
import os
import json

def merge_datasets(ground_truth_parquet: str, processed_json: str, output_json: str):
    """
    合并地面真实数据集和处理后的数据集，根据 'question' 进行匹配。

    Args:
        ground_truth_parquet (str): 地面真实数据集的Parquet文件路径。
        processed_json (str): 处理后的数据集JSON文件路径。
        output_json (str): 合并后数据集的输出JSON文件路径。
    """
    print("开始合并数据集...")

    # 1. 检查文件是否存在
    if not os.path.exists(ground_truth_parquet):
        print(f"地面真实数据集文件未找到: {ground_truth_parquet}")
        return
    if not os.path.exists(processed_json):
        print(f"处理后的数据集文件未找到: {processed_json}")
        return

    print("输入文件存在，继续读取...")

    # 2. 读取地面真实数据集
    try:
        ground_truth_df = pd.read_parquet(ground_truth_parquet)
        print(f"已加载地面真实数据集，共 {len(ground_truth_df)} 条记录。")
        print("地面真实数据集的列：", ground_truth_df.columns.tolist())
        print("地面真实数据集的前几行：")
        print(ground_truth_df.head())
    except Exception as e:
        print(f"读取地面真实数据集时出错: {e}")
        return

    # 3. 重命名 'query' 为 'question'
    if 'query' in ground_truth_df.columns:
        ground_truth_df = ground_truth_df.rename(columns={'query': 'question'})
        print("已重命名 'query' 为 'question'。")
    else:
        print("地面真实数据集中未找到 'query' 列。")
        return

    # 4. 读取处理后的数据集
    try:
        # 尝试以JSON Lines格式读取
        processed_df = pd.read_json(processed_json, lines=True)
        print("以JSON Lines格式读取处理后的数据集成功。")
    except ValueError as e:
        print(f"以JSON Lines格式读取失败: {e}")
        try:
            # 尝试以标准JSON格式读取
            processed_df = pd.read_json(processed_json, lines=False)
            print("以标准JSON格式读取处理后的数据集成功。")
        except ValueError as e:
            print(f"以标准JSON格式读取失败: {e}")
            return

    print(f"已加载处理后的数据集，共 {len(processed_df)} 条记录。")
    print("处理后的数据集的列：", processed_df.columns.tolist())
    print("处理后的数据集的前几行：")
    print(processed_df.head())

    # 5. 确保 'question' 在两个数据集中存在
    if 'question' not in ground_truth_df.columns:
        print("地面真实数据集中未找到 'question' 列。")
        return
    if 'question' not in processed_df.columns:
        print("处理后的数据集中未找到 'question' 列。")
        return

    # 6. 统一 'question' 列的格式
    ground_truth_df['question'] = ground_truth_df['question'].astype(str).str.strip().str.lower()
    processed_df['question'] = processed_df['question'].astype(str).str.strip().str.lower()
    print("已统一 'question' 列的格式（去除前后空格，转换为小写）。")

    # 7. 确认 'question' 列的唯一性
    if not ground_truth_df['question'].is_unique:
        print("地面真实数据集的 'question' 列中存在重复项。")
    if not processed_df['question'].is_unique:
        print("处理后的数据集的 'question' 列中存在重复项。")

    # 8. 合并两个数据集，基于 'question' 列
    try:
        merged_df = pd.merge(
            processed_df,
            ground_truth_df[['question', 'answer']],
            on='question',
            how='left'
        )
        print(f"合并完成，共 {len(merged_df)} 条记录。")
    except Exception as e:
        print(f"合并数据集时出错: {e}")
        return

    # 9. 检查是否有未匹配的记录
    unmatched = merged_df['answer'].isnull().sum()
    if unmatched > 0:
        print(f"警告: 有 {unmatched} 条记录未找到匹配的地面真实答案。")
        # 可选：输出未匹配的记录
        unmatched_df = merged_df[merged_df['answer'].isnull()]
        print("未匹配的记录示例：")
        print(unmatched_df[['question']].head())
    else:
        print("所有记录均找到匹配的地面真实答案。")

    # 10. 保存合并后的数据集
    try:
        merged_df.to_json(output_json, orient='records', lines=True, force_ascii=False)
        print(f"合并后的数据集已保存到: {output_json}")
    except Exception as e:
        print(f"保存合并后的数据集时出错: {e}")

if __name__ == "__main__":
    # 定义文件路径（请根据实际情况修改路径）
    ground_truth_parquet = "/Users/chengze/Desktop/MultiHopRAG_375_sampled.parquet"
    processed_json = "/Users/chengze/Desktop/query_type_HyDE_MultiHop.json"
    output_json = "/Users/chengze/Desktop/Merged_HyDE_MultiHop_with_gt.json"

    # 执行合并
    merge_datasets(ground_truth_parquet, processed_json, output_json)
