# import joblib
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import ast
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# # 读取CSV文件
# file_path = 'data/stratified_taxi_sample.csv'
# df = pd.read_csv(file_path)
# df.sort_values(by=['trip_start_time'], ascending=True, inplace=True)
#
# # 将字符串格式的时间转换为datetime对象
# df['trip_start_time'] = pd.to_datetime(df['trip_start_time'])
# df['trip_end_time'] = pd.to_datetime(df['trip_end_time'])
#
# # 提取时间特征
# df['month'] = df['trip_start_time'].dt.month
# df['day'] = df['trip_start_time'].dt.day
# df['start_hour'] = df['trip_start_time'].dt.hour
# df['start_minute'] = df['trip_start_time'].dt.minute
# df['start_second'] = df['trip_start_time'].dt.second
#
# df['end_hour'] = df['trip_end_time'].dt.hour
# df['end_minute'] = df['trip_end_time'].dt.minute
# df['end_second'] = df['trip_end_time'].dt.second
#
# # 周期性特征处理
# df['start_hour_sin'] = np.sin(2 * np.pi * df['start_hour'] / 24)
# df['start_hour_cos'] = np.cos(2 * np.pi * df['start_hour'] / 24)
# df['start_minute_sin'] = np.sin(2 * np.pi * df['start_minute'] / 60)
# df['start_minute_cos'] = np.cos(2 * np.pi * df['start_minute'] / 60)
# df['start_second_sin'] = np.sin(2 * np.pi * df['start_second'] / 60)
# df['start_second_cos'] = np.cos(2 * np.pi * df['start_second'] / 60)
#
# df['end_hour_sin'] = np.sin(2 * np.pi * df['end_hour'] / 24)
# df['end_hour_cos'] = np.cos(2 * np.pi * df['end_hour'] / 24)
# df['end_minute_sin'] = np.sin(2 * np.pi * df['end_minute'] / 60)
# df['end_minute_cos'] = np.cos(2 * np.pi * df['end_minute'] / 60)
# df['end_second_sin'] = np.sin(2 * np.pi * df['end_second'] / 60)
# df['end_second_cos'] = np.cos(2 * np.pi * df['end_second'] / 60)
# df['weekday_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
# df['weekday_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
#
# # 数据集划分
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
#
# # 保存原始训练集和测试集
# original_train_file_path = 'data/original_train_data.csv'
# original_test_file_path = 'data/original_test_data.csv'
# train_data.to_csv(original_train_file_path, index=False)
# test_data.to_csv(original_test_file_path, index=False)
# print(f"原始训练集保存到 {original_train_file_path}")
# print(f"原始测试集保存到 {original_test_file_path}")
#
# # 对训练集的时间特征进行标准化
# scaler_time = StandardScaler()
# train_data[['month', 'day']] = scaler_time.fit_transform(train_data[['month', 'day']])
#
# # 保存标准化参数
# joblib.dump(scaler_time, 'parameter/standard_scaler_time.pkl')
#
# # 处理轨迹数据
# train_data['traj'] = train_data['traj'].apply(ast.literal_eval)
#
# # 归一化动态价格系数
# scaler_dp = MinMaxScaler(feature_range=(0,1))
# scaler_dp_cur = MinMaxScaler(feature_range=(0,1))
#
# train_data[['dp_cur', 'dp_30min_prev']] = scaler_dp.fit_transform(train_data[['dp_cur', 'dp_30min_prev']])
# train_data['dp_cur'] = scaler_dp_cur.fit_transform(train_data[['dp_cur']])
#
# # 保存归一化参数
# joblib.dump(scaler_dp_cur, 'parameter/scaler_dp_cur.pkl')
# joblib.dump(scaler_dp, 'parameter/minmax_scaler_dp.pkl')
#
# # 定义轨迹标准化函数
# def normalize_trajectory(traj):
#     traj = np.array(traj)
#     centroid = np.mean(traj, axis=0)
#     normalized_traj = traj - centroid
#     max_offset = np.max(np.abs(normalized_traj))
#     scaled_traj = normalized_traj / max_offset
#     return scaled_traj.tolist(), max_offset, centroid.tolist()
#
# # 对训练集的轨迹进行标准化处理
# train_data[['traj', 'scale_factor', 'centroid']] = train_data['traj'].apply(lambda x: pd.Series(normalize_trajectory(x)))
#
# # 保存处理后的训练集
# output_file_path_train = 'data/processed_train_data.csv'
# train_data.to_csv(output_file_path_train, index=False)
#
# print(f"训练集处理完成并保存到 {output_file_path_train}")
#
# # 使用训练集的标准化参数处理测试集
# test_data[['month', 'day']] = scaler_time.transform(test_data[['month', 'day']])
# test_data[['dp_cur', 'dp_30min_prev']] = scaler_dp.transform(test_data[['dp_cur', 'dp_30min_prev']])
# test_data['dp_cur'] = scaler_dp_cur.transform(test_data[['dp_cur']])
#
# # 对测试集的轨迹进行标准化处理
# test_data['traj'] = test_data['traj'].apply(ast.literal_eval)
# test_data[['traj', 'scale_factor', 'centroid']] = test_data['traj'].apply(lambda x: pd.Series(normalize_trajectory(x)))
#
# # 保存处理后的测试集
# output_file_path_test = 'data/processed_test_data.csv'
# test_data.to_csv(output_file_path_test, index=False)
#
# print(f"测试集处理完成并保存到 {output_file_path_test}")

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ast
import numpy as np
from sklearn.model_selection import train_test_split

# 读取CSV文件
file_path = 'filtered_data.csv'
df = pd.read_csv(file_path)
df.sort_values(by=['trip_start_time'], ascending=True, inplace=True)

# 将字符串格式的时间转换为datetime对象
df['trip_start_time'] = pd.to_datetime(df['trip_start_time'])
df['trip_end_time'] = pd.to_datetime(df['trip_end_time'])

# 提取时间特征
df['month'] = df['trip_start_time'].dt.month
df['day'] = df['trip_start_time'].dt.day
df['start_hour'] = df['trip_start_time'].dt.hour
df['start_minute'] = df['trip_start_time'].dt.minute
df['start_second'] = df['trip_start_time'].dt.second

df['end_hour'] = df['trip_end_time'].dt.hour
df['end_minute'] = df['trip_end_time'].dt.minute
df['end_second'] = df['trip_end_time'].dt.second

# 周期性特征处理
df['start_hour_sin'] = np.sin(2 * np.pi * df['start_hour'] / 24)
df['start_hour_cos'] = np.cos(2 * np.pi * df['start_hour'] / 24)
df['start_minute_sin'] = np.sin(2 * np.pi * df['start_minute'] / 60)
df['start_minute_cos'] = np.cos(2 * np.pi * df['start_minute'] / 60)
df['start_second_sin'] = np.sin(2 * np.pi * df['start_second'] / 60)
df['start_second_cos'] = np.cos(2 * np.pi * df['start_second'] / 60)

df['end_hour_sin'] = np.sin(2 * np.pi * df['end_hour'] / 24)
df['end_hour_cos'] = np.cos(2 * np.pi * df['end_hour'] / 24)
df['end_minute_sin'] = np.sin(2 * np.pi * df['end_minute'] / 60)
df['end_minute_cos'] = np.cos(2 * np.pi * df['end_minute'] / 60)
df['end_second_sin'] = np.sin(2 * np.pi * df['end_second'] / 60)
df['end_second_cos'] = np.cos(2 * np.pi * df['end_second'] / 60)
df['weekday_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['dest_cluster'] = df['dest_cluster']-1
# # 数据集划分
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
split_index = int(len(df) * 0.8)
train_data = df[:split_index]
test_data = df[split_index:]
# 保存原始训练集和测试集
desired_columns_original = [
    'month', 'day', 'car_id', 'trip_start_time', 'trip_end_time',
    'slo', 'sla', 'ela', 'elo', 'day_of_week', 'daytype', 'traj',
    'dp_cur', 'dp_30min_prev', 'dp_1h_prev','weather_0', 'weather_1', 'weather_2',
    'weather_3', 'weather_4', 'weather_5', 'dest_cluster'
]
train_data_original = train_data[desired_columns_original]
test_data_original = test_data[desired_columns_original]
dp_cols = ["dp_cur", "dp_30min_prev", "dp_1h_prev"]
original_train_file_path = 'data/original_train_data.csv'
original_test_file_path = 'data/original_test_data.csv'
train_data_original.to_csv(original_train_file_path, index=False)
test_data_original.to_csv(original_test_file_path, index=False)
print(f"原始训练集保存到 {original_train_file_path}")
print(f"原始测试集保存到 {original_test_file_path}")

# 对训练集的时间特征进行标准化
scaler_time = StandardScaler()
train_data[['month', 'day']] = scaler_time.fit_transform(train_data[['month', 'day']])

# 保存标准化参数
joblib.dump(scaler_time, 'parameter/standard_scaler_time.pkl')

# 处理轨迹数据
train_data['traj'] = train_data['traj'].apply(ast.literal_eval)

# # 归一化动态价格系数
# scaler_dp = MinMaxScaler(feature_range=(0, 1))
# scaler_dp_cur = MinMaxScaler(feature_range=(0, 1))
#
# train_data[['dp_cur', 'dp_30min_prev']] = scaler_dp.fit_transform(train_data[['dp_cur', 'dp_30min_prev']])
# train_data['dp_cur'] = scaler_dp_cur.fit_transform(train_data[['dp_cur']])
#
# # 保存归一化参数
# joblib.dump(scaler_dp_cur, 'parameter/scaler_dp_cur.pkl')
# joblib.dump(scaler_dp, 'parameter/minmax_scaler_dp.pkl')

scaler_dp = MinMaxScaler(feature_range=(0, 1))
train_data[dp_cols] = scaler_dp.fit_transform(train_data[dp_cols])

# 保存 scaler（只需要一个）
joblib.dump(scaler_dp, 'parameter/minmax_scaler_dp_all.pkl')
# 定义轨迹标准化函数
def normalize_trajectory(traj):
    traj = np.array(traj)
    centroid = np.mean(traj, axis=0)
    normalized_traj = traj - centroid
    max_offset = np.max(np.abs(normalized_traj))
    scaled_traj = normalized_traj / max_offset
    return scaled_traj.tolist(), max_offset, centroid.tolist()

# 对训练集的轨迹进行标准化处理
train_data[['traj', 'scale_factor', 'centroid']] = train_data['traj'].apply(
    lambda x: pd.Series(normalize_trajectory(x))
)

# 保存处理后的训练集
desired_columns_processed = [
    'month', 'day', 'traj', 'daytype', 'dp_cur', 'dp_30min_prev','dp_1h_prev',
    'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4',
    'weather_5', 'start_hour_sin', 'start_hour_cos', 'start_minute_sin',
    'start_minute_cos', 'start_second_sin', 'start_second_cos',
    'end_hour_sin', 'end_hour_cos', 'end_minute_sin', 'end_minute_cos',
    'end_second_sin', 'end_second_cos', 'weekday_sin', 'weekday_cos',
    'scale_factor', 'centroid','dest_cluster'
]
train_data_processed = train_data[desired_columns_processed]

output_file_path_train = 'data/processed_train_data_new.csv'
train_data_processed.to_csv(output_file_path_train, index=False)

print(f"训练集处理完成并保存到 {output_file_path_train}")

# 使用训练集的标准化参数处理测试集
test_data[['month', 'day']] = scaler_time.transform(test_data[['month', 'day']])
# test_data[['dp_cur', 'dp_30min_prev']] = scaler_dp.transform(test_data[['dp_cur', 'dp_30min_prev']])
# test_data['dp_cur'] = scaler_dp_cur.transform(test_data[['dp_cur']])
# =====================
# 测试集 dp 使用训练集 scaler
# =====================
test_data[dp_cols] = scaler_dp.transform(test_data[dp_cols])

# 对测试集的轨迹进行标准化处理
test_data['traj'] = test_data['traj'].apply(ast.literal_eval)
test_data[['traj', 'scale_factor', 'centroid']] = test_data['traj'].apply(
    lambda x: pd.Series(normalize_trajectory(x))
)

# 保存处理后的测试集
test_data_processed = test_data[desired_columns_processed]

output_file_path_test = 'data/processed_test_data_new.csv'
test_data_processed.to_csv(output_file_path_test, index=False)

print(f"测试集处理完成并保存到 {output_file_path_test}")

# # **新增代码开始**
#
# # 1. 从 test_data_processed 中提取所需的列，并进行列重命名
# processed_columns = {
#     'start_hour_sin': 'hour_sin',
#     'start_hour_cos': 'hour_cos',
#     'dp_cur': 'dp_cur',
#     'dp_30min_prev': 'dp_30min_prev'
# }
#
# # 重命名列
# test_data_processed_renamed = test_data_processed.rename(columns=processed_columns)
#
# # 2. 将处理后的列添加到 test_data_original 中，对于重名列进行覆盖
# for processed_col, original_col in processed_columns.items():
#     test_data_original[original_col] = test_data_processed_renamed[original_col].values
#
# # 3. 添加新的列（hour_sin 和 hour_cos）
# test_data_original['hour_sin'] = test_data_processed_renamed['hour_sin'].values
# test_data_original['hour_cos'] = test_data_processed_renamed['hour_cos'].values
#
# # 4. 保存更新后的原始测试集
# updated_original_test_file_path = 'data/original_test_data_with_processed_columns.csv'
# test_data_original.to_csv(updated_original_test_file_path, index=False)
#
# print(f"更新后的原始测试集已保存到 {updated_original_test_file_path}")