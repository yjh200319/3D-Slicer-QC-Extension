import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 添加项目根目录到路
import numpy as np
import torch
from haarpsi import haarpsi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_3D_MSE(input_dwi, output_dwi):
    input_dwi = input_dwi.squeeze().cpu().numpy()  # [batch_size,1,128,128,128] -> [b,128,128,128]
    output_dwi = output_dwi.squeeze().cpu().numpy()  # [batch_size,1,128,128,128] -> [b,128,128,128]
    mse = np.mean((input_dwi - output_dwi) ** 2)
    return mse
def calculate_3D_LPIPS(loss_fn,input_dwi, output_dwi):
    # 加载LPIPS模型
    slices1 = input_dwi.permute(4, 0, 1, 2, 3).squeeze(1)  # [b,1,112,112,50]->[50,b,1,112,112]
    slices2 = output_dwi.permute(4, 0, 1, 2, 3).squeeze(1)
    # 应该变成并行计算 比如改成一个tensor A:[50,1,112,112] B:[50,1,112,112]
    # 计算每对切片的LPIPS相似性评分 然后计算 C = loss_fn(B) ,然后求这个的均值 np.mean(C)
    # 计算LPIPS相似性评分
    score = loss_fn(slices1, slices2)

    # 计算所有切片的平均相似性评分
    average_score = torch.mean(score).item()
    # print(f'Average LPIPS score for 3D MRI images: {average_score}')
    return average_score


def calculate_3D_HaarPSI(input_data, denoise_data):
    """
    input_data 和 denoise_data 的形状: [batch_size, 1, 128, 128, 128]
    """

    depth = input_data.size(4)
    loss = 0.0
    valid_slices = 0  # 用于计数有效切片（非全零切片）

    all_input_slices = []
    all_denoise_slices = []

    # 对深度维度进行切片
    for d in range(depth):
        input_slices = input_data[:, :, :, :, d]  # [batch_size, 128, 128] -> [batch_size,1,128,128,128]
        denoise_slices = denoise_data[:, :, :, :, d]  # [batch_size, 128, 128] ->[batch_size,1,128,128]

        # 检查切片是否为全零（即所有像素值为零）
        # non_zero_mask = ~((input_slices == 0).view(input_slices.size(0), -1).all(dim=1))
        non_zero_mask = ~((input_slices == 0).reshape(input_slices.size(0), -1).all(dim=1))
        valid_input_slices = input_slices[non_zero_mask]  # shape [16,1,128,128]
        valid_denoise_slices = denoise_slices[non_zero_mask]

        if valid_input_slices.size(0) > 0:  # 确保存在有效切片
            # 对每个有效切片进行归一化 (最小值为 0，因此仅除以最大值)
            max_val_input = valid_input_slices.reshape(valid_input_slices.size(0), -1).max(dim=1, keepdim=True)[0]
            valid_input_slices = valid_input_slices / (max_val_input.reshape(-1, 1, 1, 1) + 1e-6)

            max_val_denoise = valid_denoise_slices.reshape(valid_denoise_slices.size(0), -1).max(dim=1, keepdim=True)[
                0]
            valid_denoise_slices = valid_denoise_slices / (max_val_denoise.reshape(-1, 1, 1, 1) + 1e-6)

            all_input_slices.append(valid_input_slices)  # list shape [ [16,1,128,128],[16,128,128]]
            all_denoise_slices.append(valid_denoise_slices)
            valid_slices += valid_input_slices.size(0)  # 更新有效切片计数

    if len(all_input_slices) > 0:
        all_input_slices = torch.cat(all_input_slices, dim=0)  # 合并所有有效切片 [2048,1,128,128]
        all_denoise_slices = torch.cat(all_denoise_slices, dim=0)  # 合并所有有效切片
        # 计算 HaarPSI 损失 (假设 HaarPSI 支持批量操作)
        (psi_loss, _, _) = haarpsi(all_input_slices, all_denoise_slices, 5, 5.8)
        loss = psi_loss.mean().item()  # HaarPSI，用于最小化损失
        print("loss", loss)

    return loss
