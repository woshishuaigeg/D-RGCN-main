import os
import pandas as pd

# 定义CSV文件所在的根文件夹路径
root_folder = r'result-dgcn'

# 定义要保存结果的CSV文件路径
output_csv = r'result-dgcn.csv'

# 定义需要计算平均值的指标
metrics = ['F1', 'precision', 'recall', 'AUC', 'Accuracy', 'MCC']

# 初始化一个字典来累积指标的总和和计数
metrics_sum = {metric: 0.0 for metric in metrics}
metrics_count = {metric: 0 for metric in metrics}

# 遍历根文件夹下的所有子文件夹
for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.csv') and 'gcn' in file:
            file_path = os.path.join(subdir, file)

            # 读取当前CSV文件
            df = pd.read_csv(file_path, index_col=0)

            # 计算每个指标的平均值并累加总和和计数
            for metric in metrics:
                if metric in df.index:
                    metrics_sum[metric] += df.loc[metric, 'avg']
                    metrics_count[metric] += 1

# 计算平均值
metrics_average = {metric: metrics_sum[metric] / metrics_count[metric] if metrics_count[metric] > 0 else 0.0 for metric
                   in metrics}

# 创建保存结果的DataFrame
average_df = pd.DataFrame([metrics_average])

# 保存结果到CSV文件
average_df.to_csv(output_csv, index=False)
print(f"计算结果已保存到 {output_csv}")
