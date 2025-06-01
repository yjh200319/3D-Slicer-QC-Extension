import time
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from read_dwi import DWI_Dataset
from model import DenseNet3D
from os.path import join
import os



# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
num_gpu = torch.cuda.device_count()


def tester(test_dataloader):
    model.eval()
    volume_classify_results_list = []
    with torch.no_grad():
        for batch_idx1, (data, volume_idx) in enumerate(tqdm(test_dataloader)):
            data = data.to(device)
            # 转换为float32
            # data1 = data1.to(torch.float32)

            data[data < 0] = 0
            y_pred = model(data)

            # 验证准确率计算
            y_pred_prob = torch.sigmoid(y_pred)
            y_pred_class = (y_pred_prob >= 0.5).float().tolist()  # 每个样本的预测结果

            combined = list(zip(volume_idx.tolist(), y_pred_class))
            volume_classify_results_list.append(combined)

    print("volume classify")
    print(volume_classify_results_list)
    flattened_list = [(x, int(y[0])) for sublist in volume_classify_results_list for x, y in sublist]
    for i in range(len(flattened_list)):
        print(f"Volume idx: {flattened_list[i][0]}, class: {flattened_list[i][1]}")


def load_3DQCNet_model(model_path):
    model = DenseNet3D(num_classes=1)
    if num_gpu > 1:
        # 多卡加载：用 DataParallel 包装模型
        print("Use multi-card loading")
        model = nn.DataParallel(model)
    else:
        # 单卡加载
        print("Use single card loading")
    # 加载 checkpoint，注意 map_location 指定当前设备
    checkpoint = torch.load(model_path, map_location=device)
    # 如果是单卡加载，但模型是在多卡训练时保存的（即 checkpoint 中 key 带 "module." 前缀），需要去除前缀
    if num_gpu == 1:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            # 去掉 "module." 前缀（如果存在）
            name = k.replace('module.', '')
            new_state_dict[name] = v
        checkpoint = new_state_dict

    # 加载模型参数
    model.load_state_dict(checkpoint)
    model.to(device)

    return model


# test_single_sub.py
import time
from torch.utils.data import DataLoader
from dataset import DWI_Dataset
from model import load_3DQCNet_model

def test_single_sub(dwi_path, mask_path, bval_path, bvec_path, model_dir, test_batch_size=2):
    start_time = time.time()

    # 初始化模型
    device_id = [0, 1]  # GPU 设备
    model = load_3DQCNet_model(model_path=model_dir)

    # 加载测试数据集
    test_set = DWI_Dataset(
        dwi_path=dwi_path,
        mask_path=mask_path,
        bval_path=bval_path,
        bvec_path=bvec_path
    )

    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    print("volume len:", len(test_set))
    print("****" * 3 + " Finished loading validate data... " + "****" * 3)

    # 测试模型
    tester(test_loader)

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Time: {:.4f} minutes".format(duration_minutes))




def test_single_sub(dwi_path, mask_path, bval_path, bvec_path, model_dir, test_batch_size=2):
    start_time = time.time()

    # 初始化模型
    device_id = [0, 1]  # GPU 设备
    model = load_3DQCNet_model(model_path=model_dir)

    # 加载测试数据集
    test_set = DWI_Dataset(
        dwi_path=dwi_path,
        mask_path=mask_path,
        bval_path=bval_path,
        bvec_path=bvec_path
    )

    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    print("volume len:", len(test_set))
    print("****" * 3 + " Finished loading validate data... " + "****" * 3)

    # 测试模型
    tester(test_loader)

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Time: {:.4f} minutes".format(duration_minutes))






# if __name__ == '__main__':
#     # 计时
#     start_time = time.time()
#     # 加载数据集路径
#     dwi_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi.nii.gz'
#     mask_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi_mask.nii.gz'
#     bval_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi.bval'
#     bvec_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi.bvec'
#
#     model_dir = './checkpoints/3D_QCNet_36.pth'
#     test_batch_size = 2
#     # 超参数定义
#     # 初始化模型
#     device_id = [0, 1]
#     model = load_3DQCNet_model(model_path=model_dir)
#
#     test_set = DWI_Dataset(dwi_path=dwi_path,
#                            mask_path=mask_path,
#                            bval_path=bval_path,
#                            bvec_path=bvec_path)
#
#     test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
#     print("测试集大小volume数量:", len(test_set))
#     print("****" * 3 + "Finished loading validate data..." + "****" * 3)
#
#     # 测试模型
#     tester(test_loader)
#
#     # 可视化曲线
#     # visualize(loss_train, loss_val)
#     end_time = time.time()
#     duration_minutes = (end_time - start_time) / 60
#     print("Time: {:.4f} minutes".format(duration_minutes))
