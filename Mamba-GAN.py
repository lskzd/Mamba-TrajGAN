# 导入必要的库
import math
import os
import time
import random
import numpy as np
import gc
import torch
from opacus import PrivacyEngine
import wandb

from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.nn import Transformer
import torch.optim as optim
import pandas as pd
from mamba_ssm import Mamba2
import json
import torch.nn.functional as F


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

wandb.login(key="0c7ece70d87bb4788bd1f6fdd00fa6b2d2cdd079")

# 数据集定义
class TaxiTrajectoryDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data['traj'] = self.data['traj'].apply(json.loads)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        traj = torch.tensor(row['traj'], dtype=torch.float32)
        features = torch.tensor(row[['month', 'day', 'daytype', 'weather_0', 'weather_1',
                                     'weather_2', 'weather_3', 'weather_4', 'weather_5', 'start_hour_sin',
                                     'start_hour_cos', 'start_minute_sin', 'start_minute_cos', 'start_second_sin',
                                     'start_second_cos', 'end_hour_sin', 'end_hour_cos', 'end_minute_sin',
                                     'end_minute_cos', 'end_second_sin', 'end_second_cos', 'weekday_sin',
                                     'weekday_cos']].astype(float).values, dtype=torch.float32)
        dp_cur = torch.tensor(row['dp_cur'], dtype=torch.float32)
        dp_30min_prev = torch.tensor(row['dp_30min_prev'], dtype=torch.float32)

        return traj, features, dp_cur, dp_30min_prev


def collate_fn(batch):
    trajs, features, dp_curs, dp_30min_prevs = zip(*batch)
    trajs_padded = pad_sequence(trajs, batch_first=True, padding_value=0)
    features_stacked = torch.stack(features)
    dp_curs_stacked = torch.stack(dp_curs)
    dp_30min_prevs_stacked = torch.stack(dp_30min_prevs)
    lengths = torch.tensor([len(traj) for traj in trajs])
    mask = (trajs_padded != 0).all(dim=2).float()

    return trajs_padded, features_stacked, dp_curs_stacked, dp_30min_prevs_stacked, lengths, mask


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

#
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
#         # 初始化所有线性层的权重
#         self.apply(lambda m: initialize_weights_xavier(m))
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
#         # # 构建索引，提取每个序列的最后一个有效时间步的输出
#         # idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.size(
#         #     2)).to(output.device)  # [batch_size, 1, hidden_size]
#         # last_hidden_state = output.gather(1, idx).squeeze(1)  # [batch_size, hidden_size]
#         #
#         # # 计算动态价格系数输出
#         # dp_cur_output = self.output_activation_dp(self.dp_cur_decoder(last_hidden_state)).squeeze(-1)
#         # dp_30min_prev_output = self.output_activation_dp(self.dp_prev_decoder(last_hidden_state)).squeeze(-1)
#         # 构建索引，提取每个序列的最后一个有效时间步的输出
#         batch_size = output.size(0)
#         batch_indices = torch.arange(batch_size).to(output.device)
#         time_indices = (lengths - 1).clamp(min=0).to(output.device)
#         last_hidden_state = output[batch_indices, time_indices, :]  # [batch_size, hidden_size]
#
#         # 计算动态价格系数输出
#         dp_cur_output = self.output_activation_dp(self.dp_cur_decoder(last_hidden_state)).squeeze(-1)
#         dp_30min_prev_output = self.output_activation_dp(self.dp_prev_decoder(last_hidden_state)).squeeze(-1)
#
#         return traj_output, dp_cur_output, dp_30min_prev_output
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim  # 噪声向量的维度
        self.mamba = Mamba2(d_model=256, d_state=128, d_conv=4, expand=2)
        self.fusion = FeatureFusionLayer( 27 + noise_dim, 256)
        self.output_activation_traj = nn.Tanh()
        self.output_activation_dp = nn.Sigmoid()
        self.traj_decoder = nn.Linear(256, 2)
        self.dp_cur_decoder = nn.Linear(256, 1)
        self.dp_prev_decoder = nn.Linear(256, 1)
        # 初始化所有线性层的权重
        self.apply(lambda m: initialize_weights_xavier(m))

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

from opacus.layers import DPLSTM


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lstm = DPLSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 1)
        self.fusion = FeatureFusionLayer(27, 256)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(lambda m: initialize_weights_xavier(m))

    def forward(self, traj, features, dp_cur, dp_30min_prev, lengths):

        # 确保 dp_cur 和 dp_30min_prev 是 1D 张量
        if dp_cur.dim() == 2:
            dp_cur = dp_cur.squeeze(1)  # 从 (batch_size, 1) 转换为 (batch_size,)
        if dp_30min_prev.dim() == 2:
            dp_30min_prev = dp_30min_prev.squeeze(1)
        dp_cur_expanded = dp_cur.unsqueeze(1).unsqueeze(2).repeat(1, traj.shape[1], 1)
        dp_30min_prev_expanded = dp_30min_prev.unsqueeze(1).unsqueeze(2).repeat(1, traj.shape[1], 1)

        features_expanded = features.unsqueeze(1).repeat(1, traj.shape[1], 1)
        combined_input = torch.cat((traj, dp_cur_expanded, dp_30min_prev_expanded, features_expanded), dim=2)

        combined_input = self.dropout(combined_input)  # 添加Dropout
        # 获取设备
        device = combined_input.device
        lengths = lengths.to(device)
        # 创建掩码矩阵
        max_len = combined_input.size(1)
        mask = torch.arange(max_len).unsqueeze(0).to(lengths.device) < lengths.unsqueeze(1)
        # 扩展掩码并应用于输入
        mask_expanded = mask.unsqueeze(2).expand_as(combined_input)
        combined_input = combined_input * mask_expanded.float()
        output = self.fusion(combined_input)
        lengths = lengths.to('cpu')
        output = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        # 通过 LSTM 层（不使用 PackedSequence）
        outputs, (h_n, c_n) = self.lstm(output)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # 获取每个序列的最后一个有效时间步的输出
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(outputs.size(0), 1, outputs.size(2))
        idx = idx.to(outputs.device)
        last_outputs = outputs.gather(1, idx).squeeze(1)

        x = self.sigmoid(self.fc2(last_outputs))
        return x


class MaskedCosineLoss(nn.Module):
    def __init__(self):
        super(MaskedCosineLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, predictions, targets, mask):
        cos_sim = self.cosine_similarity(predictions, targets)
        masked_cos_sim = cos_sim * mask
        valid_elements = mask.sum()
        loss = (1 - masked_cos_sim.sum() / valid_elements) if valid_elements > 0 else torch.tensor(0.0)
        return loss


# 将模型的梯度和权重监控起来
def log_model_parameters(model, epoch):
    for name, param in model.named_parameters():
        wandb.log({f"{name}_mean": param.data.mean().item(),
                   f"{name}_std": param.data.std().item(),
                   f"{name}_min": param.data.min().item(),
                   f"{name}_max": param.data.max().item()}, step=epoch)


import torch.nn as nn


def initialize_weights_xavier(module):
    """
    使用 Xavier Uniform Initialization 初始化层权重，适用于 Tanh 和 Sigmoid 激活函数。
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    # 如果有其他类型的层，可以在这里添加相应的初始化


def initialize_weights_kaiming(module, a=0.2):
    """
    使用 Kaiming Uniform Initialization 初始化层权重，适用于 LeakyReLU 激活函数。

    参数:
    - module: 要初始化的模块。
    - a: LeakyReLU 的负斜率，默认为 0.2。
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=a, nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=a, nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    # 如果有其他类型的层，可以在这里添加相应的初始化


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs, targets, valid_sequences):
        # outputs 和 targets 的形状为 (batch_size, 1)
        # valid_sequences 为布尔张量，形状为 (batch_size,)
        outputs_valid = outputs[valid_sequences]
        targets_valid = targets[valid_sequences]
        loss = self.bce_loss(outputs_valid, targets_valid)
        return loss


def train():
    with wandb.init() as run:
        config = wandb.config
        run_id = run.id
        print(f"Run ID: {run_id}")
        # 设置随机种子
        set_seed(42)
        # 创建模型保存目录
        model_dir = f'../model_parameter/Mamba-DP-model_checkpoints/{run_id}'
        os.makedirs(model_dir, exist_ok=True)
        # print(f"Model directory: {model_dir}")
        #
        # # 打印配置以进行调试
        # print("Current configuration:")
        # print(config)
        # 加载数据集
        dataset = TaxiTrajectoryDataset('../data/processed_train_data.csv')
        loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=4,
                            pin_memory=True)

        # print("Data loader created.")
        # # 初始化模型
        generator = Generator().to('cuda')
        # discriminator = Discriminator().to('cuda')
        discriminator_private = Discriminator().to('cuda')  # 用于差分隐私训练的判别器
        # 定义优化器和损失函数
        optimizer_g = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-4)
        optimizer_d = optim.Adam(discriminator_private.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-4)
        criterion_discriminator = DiscriminatorLoss()

        criterion_generator = MaskedCosineLoss()
        criterion_dp = nn.MSELoss()
        # criterion_tv = TotalVariationLoss()
        print("Optimizers and loss functions defined.")
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config.epochs)
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config.epochs)
        # 设置差分隐私参数并将 PrivacyEngine 附加到判别器的优化器
        delta = 1e-5  # 设置 delta 值
        privacy_engine = PrivacyEngine(secure_mode=False)

        discriminator_private, optimizer_d, loader = privacy_engine.make_private(
            module=discriminator_private,
            optimizer=optimizer_d,
            data_loader=loader,
            noise_multiplier=1.5,
            max_grad_norm=1.0,
        )
        # discriminator = discriminator_private._module
        print("PrivacyEngine applied to discriminator.")

        for epoch in range(config.epochs):
            generator.train()
            # discriminator.train()
            discriminator_private.train()
            start_time = time.time()

            for trajs_padded, features, dp_curs, dp_30min_prevs, lengths, mask in loader:
                trajs_padded, features = trajs_padded.to('cuda'), features.to('cuda')
                dp_curs, dp_30min_prevs = dp_curs.to('cuda'), dp_30min_prevs.to('cuda')
                # lengths = lengths.to('cuda')

                mask = mask.to('cuda')
                valid_sequences = lengths > 0  # 布尔张量，形状为 (batch_size,)

                # 生成噪声
                noise_traj = torch.randn(trajs_padded.size(0), trajs_padded.size(1), 2).to('cuda')
                noise_dp = torch.randn(trajs_padded.size(0), trajs_padded.size(1), 1).to('cuda')
                noise_dp_30min_prev = torch.randn(trajs_padded.size(0), trajs_padded.size(1), 1).to('cuda')
                z = torch.randn(trajs_padded.size(0), generator.noise_dim).to('cuda')
                # 训练判别器
                optimizer_d.zero_grad(set_to_none=True)
                real_labels = torch.ones(trajs_padded.size(0), 1).to('cuda') * 0.9
                fake_labels = torch.zeros(trajs_padded.size(0), 1).to('cuda')

                outputs_real = discriminator_private(trajs_padded, features, dp_curs, dp_30min_prevs, lengths)
                d_loss_real = criterion_discriminator(outputs_real, real_labels, valid_sequences)
                real_score = outputs_real.mean().item()
                #
                # traj_output, dp_cur_output, dp_30min_prev_output = generator(
                #     trajs_padded, features, dp_curs, dp_30min_prevs, noise_traj, noise_dp, noise_dp_30min_prev, mask,
                #     lengths)
                traj_output, dp_cur_output, dp_30min_prev_output = generator(
                    trajs_padded, features, dp_curs, dp_30min_prevs, z,mask,
                    lengths)

                outputs_fake = discriminator_private(traj_output.detach(), features.detach(), dp_cur_output.detach(),
                                                     dp_30min_prev_output.detach(), lengths.detach())
                d_loss_fake = criterion_discriminator(outputs_fake, fake_labels, valid_sequences)
                fake_score = outputs_fake.mean().item()

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad(set_to_none=True)
                # 训练生成器
                optimizer_g.zero_grad()

                # # 冻结判别器参数以防止梯度累积
                # for param in discriminator_private.parameters():
                #     param.requires_grad = False

                # with torch.no_grad():
                outputs_gen = discriminator_private(traj_output, features, dp_cur_output, dp_30min_prev_output, lengths)
                adversarial_loss_value = criterion_discriminator(outputs_gen, real_labels, valid_sequences)
                #

                trajectory_loss_value = criterion_generator(traj_output, trajs_padded, mask)
                # tv_loss_value = criterion_tv(traj_output, mask)

                dp_cur_loss_value = criterion_dp(dp_cur_output, dp_curs)
                dp_30min_prev_loss_value = criterion_dp(dp_30min_prev_output, dp_30min_prevs)

                # 设置损失权重
                adversarial_weight = 1.0
                trajectory_weight = 10.0
                dp_cur_weight = 20.0
                dp_30min_prev_weight = 20.0
                tv_loss_weight = 5.0
                g_loss = (
                        adversarial_weight * adversarial_loss_value +
                        trajectory_weight * trajectory_loss_value +
                        dp_cur_weight * dp_cur_loss_value +
                        dp_30min_prev_weight * dp_30min_prev_loss_value
                )
                g_loss.backward()
                optimizer_g.step()

                # # 恢复判别器参数
                # for param in discriminator_private.parameters():
                #     param.requires_grad = True
            epsilon = privacy_engine.accountant.get_epsilon(delta)
            end_time = time.time()

            scheduler_g.step()
            scheduler_d.step()
            # 保存模型
            if (epoch + 1) % 1 == 0:
                torch.save(generator.state_dict(), os.path.join(model_dir, f'generator_epoch_{epoch + 1}.pth'))
                torch.save(discriminator_private.state_dict(),
                           os.path.join(model_dir, f'discriminator_epoch_{epoch + 1}.pth'))

            # 将模型的梯度和权重监控起来
            log_model_parameters(generator, epoch)
            log_model_parameters(discriminator_private, epoch)
            print(f'Epoch [{epoch + 1}/{config.epochs}], Loss D: {d_loss.item()}, Loss G: {g_loss.item()}, '
                  f'Real Score: {real_score}, Fake Score: {fake_score},  Delta: {delta} '
                  f'Adversarial Loss: {adversarial_loss_value.item()}, '
                  f'Trajectory Loss: {trajectory_loss_value.item()}, '
                  f'DP Cur Loss: {dp_cur_loss_value.item()}, '
                  f'DP 30min Prev Loss: {dp_30min_prev_loss_value.item()}, '
                  f'Time: {end_time - start_time}'
                  )

            wandb.log({
                'epoch': epoch + 1,
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item(),
                'Real Score': real_score,
                'Fake Score': fake_score,
                'Adversarial Loss': adversarial_loss_value.item(),
                'Trajectory Loss': trajectory_loss_value.item(),
                'DP Cur Loss': dp_cur_loss_value.item(),
                # 'Epsilon': epsilon,
                'Delta': delta,
                'DP 30min Prev Loss': dp_30min_prev_loss_value.item(),
                'Time': end_time - start_time
            })
            # if epsilon > 5:
            #     print(f"Epsilon {epsilon:.2f} exceeded 10, stopping training.")
            #     wandb.log({'Epsilon Exceeded 10': True, 'Stopped Epoch': epoch + 1})
            #     break  # 退出训练循环



if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'g_loss',
            'goal': 'minimize'

        },
        'parameters': {
            'batch_size': {
                'values': [1024]
            },
            # 'g_lr': {
            #     'min': 1e-5,
            #     'max': 1e-2,
            #     'distribution': 'log_uniform_values'
            # },
            # 'd_lr': {
            #     'min': 1e-5,
            #     'max': 1e-2,
            #     'distribution': 'log_uniform_values'
            # },
            # 'noise_multiplier': {
            #     'min': 0.5,
            #     'max': 2.0
            # },
            # 'max_grad_norm': {
            #     'min': 0.5,
            #     'max': 1.5
            # },
            'epochs': {
                'value': 200
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='Mamba-DP-Test')

    # sweep_id = 'h24it0i3'

    wandb.agent(sweep_id, function=train, count=1, project='Mamba-DP-Test')
