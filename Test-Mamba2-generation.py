import json

import joblib
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd
from argparse import Namespace
from mamba_ssm import Mamba2
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader
import math

configparser  = Namespace(
    epoch ='90'
)

class FeatureFusionLayer(nn.Module):
    def __init__(self, embedding_dims, hidden_dim=256):
        """
        embedding_dims: List[int]，每个嵌入的特征维度
        hidden_dim: int，Dense 层的输出维度
        """
        super(FeatureFusionLayer, self).__init__()

        # 计算拼接后的特征维度
        self.concat_dim = embedding_dims

        # 定义全连接层，将拼接后的特征映射到 hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),
            nn.LayerNorm(hidden_dim)
        )



    def forward(self, embeddings):
        """
        embeddings: List[Tensor] 或者 Tensor，形状为 [batch_size, seq_length, feature_dim_i]
        返回:
            emb_traj: Tensor，形状为 [batch_size, seq_length, hidden_dim]
        """
        if isinstance(embeddings, list):
            # 将嵌入沿着特征维度拼接
            concat_input = torch.cat(embeddings, dim=2)  # [batch, seq_length, sum(embedding_dims)]
        else:
            concat_input = embeddings  # 假设已拼接

        # # 监控数据范围
        # print(f"Before FC - min: {concat_input.min().item()}, max: {concat_input.max().item()}")

        # 通过全连接层
        emb_traj = self.fc(concat_input)  # [batch, seq_length, hidden_dim]

        # # 监控输出范围
        # print(f"After FC - min: {emb_traj.min().item()}, max: {emb_traj.max().item()}")

        return emb_traj
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.mamba = Mamba2(d_model=256, d_state=128, d_conv=4, expand=2)
#         self.fusion = FeatureFusionLayer(27, 256)
#         self.output_activation_traj = nn.Tanh()
#         self.output_activation_dp = nn.Sigmoid()
#         self.noise_fusion_traj = nn.Linear(4, 2)
#         self.noise_fusion_dp_cur = nn.Linear(2, 1)
#         self.noise_fusion_dp_30min_prev = nn.Linear(2, 1)
#         self.traj_decoder = nn.Linear(256, 2)
#         self.dp_cur_decoder = nn.Linear(256, 1)
#         self.dp_prev_decoder = nn.Linear(256, 1)
#
#
#     def forward(self, traj, features, dp_cur, dp_30min_prev, noise_traj, noise_dp, noise_dp_30min_prev, mask, lengths):
#         mask = mask.unsqueeze(2)  # [batch_size, seq_len, 1]
#         traj_noised_input = torch.cat([traj, noise_traj], dim=2)
#         traj_noised = self.noise_fusion_traj(traj_noised_input)
#         traj_noised = self.output_activation_traj(traj_noised)
#
#         dp_cur_expanded = dp_cur.unsqueeze(1).unsqueeze(2).repeat(1, traj.shape[1], 1)
#         dp_30min_prev_expanded = dp_30min_prev.unsqueeze(1).unsqueeze(2).repeat(1, traj.shape[1], 1)
#
#         noise_dp = noise_dp[:, :dp_cur_expanded.size(1), :1]
#         noise_dp_30min = noise_dp_30min_prev[:, :dp_cur_expanded.size(1), :1]
#
#         dp_cur_noised_input = torch.cat([dp_cur_expanded, noise_dp], dim=2)
#         dp_cur_noised = self.noise_fusion_dp_cur(dp_cur_noised_input)
#         dp_cur_noised = self.output_activation_dp(dp_cur_noised)
#
#         dp_30min_prev_noised_input = torch.cat([dp_30min_prev_expanded, noise_dp_30min], dim=2)
#         dp_30min_prev_noised = self.noise_fusion_dp_30min_prev(dp_30min_prev_noised_input)
#         dp_30min_prev_noised = self.output_activation_dp(dp_30min_prev_noised)
#
#         features_expanded = features.unsqueeze(1).repeat(1, traj.shape[1], 1)
#         combined_input = torch.cat((traj_noised, dp_cur_noised, dp_30min_prev_noised, features_expanded), dim=2)
#
#         combined_input = combined_input * mask.float()
#         output = self.fusion(combined_input)
#
#         output = self.mamba(output)
#
#         # 解码并应用激活函数（直接对整个序列进行）
#         traj_output = self.output_activation_traj(self.traj_decoder(output))
#
#         # 利用 mask 将无效时间步的输出置零
#         traj_output = traj_output * mask  # [batch_size, seq_len, 2]
#
#         # 构建索引，提取每个序列的最后一个有效时间步的输出
#         idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.size(
#             2)).to(output.device)  # [batch_size, 1, hidden_size]
#         last_hidden_state = output.gather(1, idx).squeeze(1)  # [batch_size, hidden_size]
#
#         # 计算动态价格系数输出
#         dp_cur_output = self.output_activation_dp(self.dp_cur_decoder(last_hidden_state)).squeeze(-1)
#         dp_30min_prev_output = self.output_activation_dp(self.dp_prev_decoder(last_hidden_state)).squeeze(-1)
#
#         return traj_output, dp_cur_output, dp_30min_prev_output

# generator_path = f'../model_parameter/Mamba-with-DP-model_checkpoints/1/generator_epoch_{configparser.epoch}.pth'
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim  # 噪声向量的维度
        self.mamba = Mamba2(d_model=256, d_state=128, d_conv=4, expand=2)
        # FeatureFusionLayer 的输入维度调整为 真实轨迹维度（2） + 原始特征维度（27） + 动态价格特征（2） + 噪声维度（noise_dim）
        self.fusion = FeatureFusionLayer( 27 + noise_dim, 256)
        self.output_activation_traj = nn.Tanh()
        self.output_activation_dp = nn.Sigmoid()
        self.traj_decoder = nn.Linear(256, 2)
        self.dp_cur_decoder = nn.Linear(256, 1)
        self.dp_prev_decoder = nn.Linear(256, 1)


    def forward(self, traj, features, dp_cur, dp_30min_prev, z, mask, lengths):
        mask = mask.unsqueeze(2)  # [batch_size, seq_len, 1]
        batch_size, seq_len = traj.size(0), traj.size(1)

        # 扩展特征到序列长度
        features_expanded = features.unsqueeze(1).repeat(1, seq_len, 1)
        dp_cur_expanded = dp_cur.unsqueeze(1).unsqueeze(2).repeat(1, seq_len, 1)
        dp_30min_prev_expanded = dp_30min_prev.unsqueeze(1).unsqueeze(2).repeat(1, seq_len, 1)

        # 扩展噪声向量到序列长度
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, noise_dim]

        # 将真实轨迹、特征、动态价格和噪声拼接在一起
        combined_input = torch.cat((traj, features_expanded, dp_cur_expanded, dp_30min_prev_expanded, z_expanded), dim=2)

        # 通过特征融合层
        combined_input = combined_input * mask.float()
        output = self.fusion(combined_input)

        # 通过 Mamba2 网络
        output = self.mamba(output)

        # 解码生成新的轨迹
        traj_output = self.output_activation_traj(self.traj_decoder(output))
        traj_output = traj_output * mask  # [batch_size, seq_len, 2]

        # 提取每个序列的最后一个有效时间步的输出，用于生成动态价格系数
        batch_indices = torch.arange(batch_size).to(output.device)
        time_indices = (lengths - 1).clamp(min=0).to(output.device)
        last_hidden_state = output[batch_indices, time_indices, :]  # [batch_size, hidden_size]

        # 生成动态价格系数
        dp_cur_output = self.output_activation_dp(self.dp_cur_decoder(last_hidden_state)).squeeze(-1)
        dp_30min_prev_output = self.output_activation_dp(self.dp_prev_decoder(last_hidden_state)).squeeze(-1)

        return traj_output, dp_cur_output, dp_30min_prev_output
# generator_path = f'../model_parameter/Mamba-DP-model_checkpoints/0xga1ls8/generator_epoch_{configparser.epoch}.pth'#无attention
generator_path = f'../model_parameter/Mamba-DP-model_checkpoints/g04ifb23/generator_epoch_{configparser.epoch}.pth'


model = Generator()
model.load_state_dict(torch.load(generator_path))
model.eval()
model.to('cuda')
data = pd.read_csv('../data/processed_test_data.csv')
# 定义数据
class TaxiTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.data['traj'] = self.data['traj'].apply(json.loads)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        traj = torch.tensor(row['traj'], dtype=torch.float32)
        features = torch.tensor(row[['month', 'day', 'daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4', 'weather_5', 'start_hour_sin', 'start_hour_cos', 'start_minute_sin', 'start_minute_cos', 'start_second_sin', 'start_second_cos', 'end_hour_sin', 'end_hour_cos', 'end_minute_sin', 'end_minute_cos', 'end_second_sin', 'end_second_cos', 'weekday_sin', 'weekday_cos']].astype(float).values, dtype=torch.float32)
        dp_cur = torch.tensor(row['dp_cur'], dtype=torch.float32)
        dp_30min_prev = torch.tensor(row['dp_30min_prev'], dtype=torch.float32)
        dest_cluster = row['dest_cluster']
        return traj, features, dp_cur, dp_30min_prev,dest_cluster

def collate_fn(batch):
    trajs, features, dp_curs, dp_30min_prevs,dest_clusters = zip(*batch)
    trajs_padded = pad_sequence(trajs, batch_first=True, padding_value=0)
    features_stacked = torch.stack(features)
    dp_curs_stacked = torch.stack(dp_curs)
    dp_30min_prevs_stacked = torch.stack(dp_30min_prevs)
    lengths = torch.tensor([len(traj) for traj in trajs])
    mask = (trajs_padded != 0).all(dim=2).float()
    dest_clusters = list(dest_clusters)
    return trajs_padded, features_stacked, dp_curs_stacked, dp_30min_prevs_stacked, lengths, mask,dest_clusters

# 加载数据
dataset = TaxiTrajectoryDataset(data)
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,shuffle=False)

def convert_to_original_format(traj_output, dp_cur_output, dp_30min_prev_output, features_output,dest_clusters):
    # 确保所有张量从 GPU 移到 CPU，使用.detach()分离梯度，并转换为 NumPy 数组
    traj_list = [traj.detach().cpu().numpy().tolist() for traj in traj_output]
    dp_cur_list = dp_cur_output.detach().cpu().numpy().tolist()
    dp_30min_prev_list = dp_30min_prev_output.detach().cpu().numpy().tolist()

    # 转换 features_output 张量到 NumPy 数组，确保维度正确
    features_array = features_output.detach().cpu().numpy()

    # 如果 features_output 是多维的，需要调整其形状以适应 DataFrame
    if features_array.ndim > 2:
        features_array = features_array.reshape(features_array.shape[0], -1)  # 可能需要根据具体情况调整

    # 创建 DataFrame
    df = pd.DataFrame(features_array, columns=[
        'month', 'day', 'daytype', 'weather_0', 'weather_1', 'weather_2', 'weather_3',
        'weather_4', 'weather_5', 'start_hour_sin', 'start_hour_cos', 'start_minute_sin',
        'start_minute_cos', 'start_second_sin', 'start_second_cos', 'end_hour_sin',
        'end_hour_cos', 'end_minute_sin', 'end_minute_cos', 'end_second_sin', 'end_second_cos',
        'weekday_sin', 'weekday_cos'
    ])
    df['traj'] = traj_list
    df['dp_cur'] = dp_cur_list
    df['dp_30min_prev'] = dp_30min_prev_list
    df['dest_cluster'] = dest_clusters

    return df
# 使用模型进行预测
generated_data_list = []

# 使用模型进行预测并收集数据
for trajs_padded, features, dp_cur, dp_30min_prev, lengths, mask,dest_cluster in loader:
    with torch.no_grad(): # 确保不计算梯度
        trajs_padded, features = trajs_padded.to('cuda'), features.to('cuda')
        dp_cur, dp_30min_prev = dp_cur.to('cuda'), dp_30min_prev.to('cuda')
        lengths = lengths.to('cuda')
        mask = mask.to('cuda')
        # 生成噪声
        noise_traj = torch.rand(trajs_padded.size(0), trajs_padded.size(1), 2).to('cuda')
        noise_dp = torch.rand(trajs_padded.size(0), trajs_padded.size(1), 1).to('cuda')
        noise_dp_30min_prev = torch.rand(trajs_padded.size(0), trajs_padded.size(1), 1).to('cuda')
        z = torch.randn(trajs_padded.size(0), model.noise_dim).to('cuda')
        # traj_output, dp_cur_output, dp_30min_prev_output = model(
        #     trajs_padded, features, dp_cur, dp_30min_prev, noise_traj, noise_dp, noise_dp_30min_prev, mask, lengths)
        traj_output, dp_cur_output, dp_30min_prev_output = model(
            trajs_padded, features, dp_cur, dp_30min_prev, z, mask,
            lengths)
        formatted_data = convert_to_original_format(traj_output, dp_cur_output, dp_30min_prev_output, features,dest_cluster)
        generated_data_list.append(formatted_data)


final_generated_data = pd.concat(generated_data_list,ignore_index=True)

final_generated_data['scale_factor'] = data['scale_factor'].values.flatten()
final_generated_data['centroid'] = data['centroid'].values.flatten()
# 保存DataFrame到CSV文件
# Load the saved scalers
scaler_time = joblib.load('../parameter/standard_scaler_time.pkl')
scaler_dp = joblib.load('../parameter/minmax_scaler_dp.pkl')

# Step 1: Inverse Transform Standardized 'month' and 'day' Columns
# ----------------------------------------------------------------

# Reverse standardization of 'month' and 'day'
final_generated_data[['month', 'day']] = scaler_time.inverse_transform(
    final_generated_data[['month', 'day']])

# Round and convert to integers
final_generated_data['month'] = final_generated_data['month'].round().astype(int)
final_generated_data['day'] = final_generated_data['day'].round().astype(int)

# Step 2: Inverse Transform Scaled 'dp_cur' and 'dp_30min_prev' Columns
# ----------------------------------------------------------------------

# Reverse scaling of 'dp_cur' and 'dp_30min_prev'
final_generated_data[['dp_cur', 'dp_30min_prev']] = scaler_dp.inverse_transform(
    final_generated_data[['dp_cur', 'dp_30min_prev']])

# Step 3: Reverse Cyclic Features to Original Time Components
# -----------------------------------------------------------

def reverse_cyclic(df, sin_col, cos_col, max_value, col_name):
    angle = np.arctan2(df[sin_col], df[cos_col])
    value = (angle % (2 * np.pi)) * (max_value / (2 * np.pi))
    df[col_name] = value.round().astype(int)
    return df

# Reverse 'start_hour', 'start_minute', 'start_second'
final_generated_data = reverse_cyclic(final_generated_data, 'start_hour_sin', 'start_hour_cos', 24, 'start_hour')
final_generated_data = reverse_cyclic(final_generated_data, 'start_minute_sin', 'start_minute_cos', 60, 'start_minute')
final_generated_data = reverse_cyclic(final_generated_data, 'start_second_sin', 'start_second_cos', 60, 'start_second')

# Reverse 'end_hour', 'end_minute', 'end_second'
final_generated_data = reverse_cyclic(final_generated_data, 'end_hour_sin', 'end_hour_cos', 24, 'end_hour')
final_generated_data = reverse_cyclic(final_generated_data, 'end_minute_sin', 'end_minute_cos', 60, 'end_minute')
final_generated_data = reverse_cyclic(final_generated_data, 'end_second_sin', 'end_second_cos', 60, 'end_second')

# Step 4: Reconstruct 'day_of_week' from Cyclic Features
# ------------------------------------------------------

final_generated_data = reverse_cyclic(final_generated_data, 'weekday_sin', 'weekday_cos', 7, 'day_of_week')
final_generated_data['day_of_week'] = final_generated_data['day_of_week'] % 7

# Step 5: Reconstruct 'trip_start_time' and 'trip_end_time'
# ---------------------------------------------------------

# Assuming the year is known or setting it to a default value
final_generated_data['year'] = 2015  # Replace with the correct year if known

# Reconstruct 'trip_start_time'
final_generated_data['trip_start_time'] = pd.to_datetime({
    'year': final_generated_data['year'],
    'month': final_generated_data['month'],
    'day': final_generated_data['day'],
    'hour': final_generated_data['start_hour'],
    'minute': final_generated_data['start_minute'],
    'second': final_generated_data['start_second']
})

# Reconstruct 'trip_end_time'
final_generated_data['trip_end_time'] = pd.to_datetime({
    'year': final_generated_data['year'],
    'month': final_generated_data['month'],
    'day': final_generated_data['day'],
    'hour': final_generated_data['end_hour'],
    'minute': final_generated_data['end_minute'],
    'second': final_generated_data['end_second']
})

# Step 6: Inverse Transform 'traj' Column
# ---------------------------------------

# Ensure 'traj' is a list of lists
final_generated_data['traj'] = final_generated_data['traj'].apply(lambda x: [list(map(float, coord)) for coord in x])
final_generated_data['centroid'] = final_generated_data['centroid'].apply(json.loads)

# Denormalize trajectory using scale_factor and centroid
def denormalize_trajectory(row):
    scaled_traj = np.array(row['traj'])
    scale_factor = row['scale_factor']
    centroid = np.array(row['centroid'])
    normalized_traj = scaled_traj * scale_factor
    original_traj = normalized_traj + centroid
    return original_traj.tolist()

if 'centroid' in final_generated_data.columns:
    final_generated_data['traj'] = final_generated_data.apply(denormalize_trajectory, axis=1)
else:
    print("Centroid information is missing. Cannot denormalize 'traj' without centroids.")
    # Alternatively, adjust your preprocessing code to save centroids

# Step 7: Save the Restored Data
# ------------------------------

# Optionally drop helper columns
final_generated_data = final_generated_data.drop(columns=[
    'start_hour_sin', 'start_hour_cos', 'start_minute_sin', 'start_minute_cos',
    'start_second_sin', 'start_second_cos', 'end_hour_sin', 'end_hour_cos',
    'end_minute_sin', 'end_minute_cos', 'end_second_sin', 'end_second_cos',
    'weekday_sin', 'weekday_cos', 'year'
])

# Save the restored data to a new CSV file
final_generated_data.to_csv(f'../result/Mamba2/Mamba2_generated_data-without-DP_{configparser.epoch}.csv', index=False)
print("Data restoration complete. Restored data saved to 'restored_generated_data.csv'.")
# final_generated_data.to_csv('generated_data_35.csv', index=False)

