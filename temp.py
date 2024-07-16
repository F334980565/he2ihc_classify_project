import pandas as pd

# 读取 CSV 文件
path = '/root/projects/wu/classify_project/probs_save/IHC_all/csv/probs.csv'
df = pd.read_csv(path)

# 替换 img_path 中的一部分字符
df['img_path'] = df['img_path'].str.replace('/home/s611/Projects', '/root/projects')

# 将结果覆盖保存到原始 CSV 文件
df.to_csv(path, index=False)