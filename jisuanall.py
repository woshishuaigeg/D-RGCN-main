import os
import pandas as pd

# 定义CSV文件所在的根文件夹路径
root_folder = r'result-dgcn'

# 定义要保存结果的CSV文件路径
output_csv = r'result_dgcn.csv'

# 初始化一个列表来存储所有文件的数据
all_data = []

# 遍历根文件夹下的所有子文件夹
for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.csv') and 'gcn' in file:
            file_path = os.path.join(subdir, file)

            # 读取当前CSV文件的第一列数据
            df = pd.read_csv(file_path, index_col=0, usecols=['avg'])

            # 重命名列为指标名称
            df = df.rename_axis('Metric').reset_index()

            # 添加文件名作为一列
            df['filename'] = file

            # 将该文件的数据添加到列表中
            all_data.append(df)

            # 添加一个空白行（值为None）以区分文件
            all_data.append(pd.DataFrame([{'Metric': None, 'avg': None, 'filename': None}]))

# 将所有数据合并为一个DataFrame
result_df = pd.concat(all_data, ignore_index=True)

# 保存结果到CSV文件
result_df.to_csv(output_csv, index=False)
print(f"所有文件的avg数据已保存到 {output_csv}")
