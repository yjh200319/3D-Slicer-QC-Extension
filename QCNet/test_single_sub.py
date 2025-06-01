import time
from collections import OrderedDict

import lpips
import numpy as np
import torch
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from QC.read_dwi import DWI_Dataset
from QC.model import DenseNet3D
from os.path import join
import os

from React_QC.model import QualityAssessmentPipeline, SVM_DataLoader
from React_QC.utils import calculate_3D_MSE, calculate_3D_HaarPSI, calculate_3D_LPIPS
from React_QC.haarpsi import haarpsi

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
num_gpu = torch.cuda.device_count()

current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir: ", current_dir)


def tester(test_dataloader, model, progress_callback=None):
    model.eval()
    volume_classify_results_list = []
    total = len(test_dataloader)

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

            if progress_callback:
                current_progress = 30 + int(70 * (batch_idx1 + 1) / total)
                progress_callback(current_progress, f"Processing volume {batch_idx1 + 1}/{total}")

    print("volume classify")
    print(volume_classify_results_list)
    flattened_list = [(x, int(y[0])) for sublist in volume_classify_results_list for x, y in sublist]
    for i in range(len(flattened_list)):
        print(f"Volume idx: {flattened_list[i][0]}, class: {flattened_list[i][1]}")

    return flattened_list


def load_3DQCNet_model(model_path):
    model = DenseNet3D(num_classes=1)
    if num_gpu > 1:
        # 多卡加载：用 DataParallel 包装模型
        print("使用多卡加载")
        model = nn.DataParallel(model)
    else:
        # 单卡加载
        print("使用单卡加载")
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


def test_single(dwi_path, mask_path, bval_path, bvec_path, model_dir, test_batch_size=2, progress_callback=None):
    start_time = time.time()

    # 初始化模型
    device_id = [0, 1]  # GPU 设备
    model = load_3DQCNet_model(model_path=model_dir)
    print("1111111")

    # 加载数据的时间进度条
    if progress_callback:
        progress_callback(20, "Loading model...")

    # 加载测试数据集
    test_set = DWI_Dataset(
        dwi_path=dwi_path,
        mask_path=mask_path,
        bval_path=bval_path,
        bvec_path=bvec_path
    )

    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    # 加载数据的时间进度条
    if progress_callback:
        progress_callback(30, "Loading dataset...")

    print("测试集大小volume数量:", len(test_set))
    print("****" * 3 + " Finished loading validate data... " + "****" * 3)

    # 测试模型
    flattened_list = tester(test_loader, model, progress_callback)

    if progress_callback:
        progress_callback(80, "Finished Testing...")
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Time: {:.4f} minutes".format(duration_minutes))

    return flattened_list


# ########################React-QC的代码#######################################


def read_bval_bvec_file(bval_path, bvec_path):
    bvals, bvecs = read_bvals_bvecs(
        bval_path,
        bvec_path
    )
    bvals = np.round(bvals / 100) * 100

    return bvals, bvecs


# 对每个volume进行预处理m,大致是先成mask然后进行归一化,然后转化成预处理的vol,并返回vol和mask
def preprocess_volume(vol, mask):
    """增强型预处理"""
    vol = vol * mask
    normalized = (vol - np.mean(vol)) / np.std(vol)
    normalized[normalized < 0] = 0
    normalized = normalized * mask
    input_vol = torch.tensor((normalized), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return input_vol, mask


def tester(test_dataloader, model, progress_callback=None):
    model.eval()
    volume_classify_results_list = []
    total = len(test_dataloader)

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

            if progress_callback:
                current_progress = 30 + int(70 * (batch_idx1 + 1) / total)
                progress_callback(current_progress, f"Processing volume {batch_idx1 + 1}/{total}")

    print("volume classify")
    print(volume_classify_results_list)
    flattened_list = [(x, int(y[0])) for sublist in volume_classify_results_list for x, y in sublist]
    for i in range(len(flattened_list)):
        print(f"Volume idx: {flattened_list[i][0]}, class: {flattened_list[i][1]}")

    return flattened_list


def load_UNet_model(model_path):
    pipeline = QualityAssessmentPipeline(model_path)
    unet_model = pipeline.load_unet(model_path['unet'])
    return unet_model


def load_SVM_model(model_path):
    pipeline = QualityAssessmentPipeline(model_path)
    svm_model = pipeline.load_svm(model_path=model_path['svm'])

    return svm_model


def test_single_by_react_qc(dwi_path, mask_path, bval_path, bvec_path, progress_callback=None):
    start_time = time.time()

    # 构造相对路径
    react_qc_dir = os.path.join(current_dir, "React_QC")
    npy_path = os.path.join(react_qc_dir, "svm_train_data")

    # npy_path = r'E:\APP\Slicer 5.8.1\QCNet\QCNet\React_QC\svm_train_data'
    svm_train_data_list = ['good.npy', 'ghost.npy', 'spike.npy', 'swap.npy',
                           'motion.npy', 'eddy.npy', 'bias.npy']
    svm_train_data_list_with_path = [os.path.join(npy_path, name) for name in svm_train_data_list]

    model_path = {
        # 'unet': r'E:\APP\Slicer 5.8.1\QCNet\QCNet\React_QC\UNet3D_model.pth',
        # 'svm': r'E:\APP\Slicer 5.8.1\QCNet\QCNet\React_QC\svm_model.pkl'
        'unet': os.path.join(react_qc_dir, "UNet3D_model.pth"),
        'svm': os.path.join(react_qc_dir, "svm_model.pkl")
    }

    unet_model = load_UNet_model(model_path)
    svm_model = load_SVM_model(model_path)

    # 初始化模型
    device_id = [0, 1]  # GPU 设备
    print("1111111")

    dwi_data, affine = load_nifti(dwi_path, return_img=False)
    mask_data, _ = load_nifti(mask_path, return_img=False)  # 一个mask用于扩展成4D
    mask, _ = load_nifti(mask_path, return_img=False)  # 一个mask用于保留成3D便于计算管理

    # 得到的bvals是经过预处理的除以100*100以后的整数bval
    bvals, bvecs = read_bval_bvec_file(bval_path, bvec_path)
    mask_data = np.expand_dims(mask_data, axis=-1)
    dwi_data = dwi_data * mask_data

    # 这里获取非0的bval的索引
    non_b0_indices = np.where(bvals != 0)[0]

    # 记录3个指标features
    features = []

    loss_fn = lpips.LPIPS(net='alex').to(device)  # [batch,c,112,112,50] ->[50,batch,c,112,112]
    total = len(non_b0_indices)
    unet_model.eval()
    with torch.no_grad():
        for i, vol_idx in enumerate(non_b0_indices):
            # 这里是按照volume的顺序进行测试的,即non_b0_indices测试的,所以应该不用去记住名字
            # 这里的vol_idx和i其实能够对应上
            # 数据预处理
            vol1, mask1 = preprocess_volume(dwi_data[..., vol_idx], mask)
            vol1 = vol1.to(device)
            mask1 = mask1.to(device)
            # 模型推理
            corrected_vol = unet_model(vol1)
            corrected_vol = corrected_vol * mask1
            corrected_vol[corrected_vol < 0] = 0

            # ####### 伪影校正替代,这里并非替代,应该是先存储起来,如果有问题了再替换

            # ########
            print("vol.shape:", vol1.shape)
            print("corrected.shape:", corrected_vol.shape)
            # 特征计算
            metrics = {
                'mse': calculate_3D_MSE(vol1, corrected_vol),
                'lpips': calculate_3D_LPIPS(loss_fn=loss_fn, input_dwi=vol1, output_dwi=corrected_vol),
                'haarpsi': calculate_3D_HaarPSI(input_data=vol1, denoise_data=corrected_vol)
            }

            corrected_vol = corrected_vol.squeeze()
            if corrected_vol.is_cuda:
                corrected_vol = corrected_vol.cpu()
            corrected_vol = corrected_vol.detach().numpy()
            features.append([metrics[k] for k in ['mse', 'lpips', 'haarpsi']])

            if progress_callback:
                current_progress = 10 + int(70 * (i + 1) / total)
                progress_callback(current_progress, f"Processing volume {i + 1}/{total}")

    # Unet评估完以后开始SVM的评估
    data_loader = SVM_DataLoader(train_paths=svm_train_data_list_with_path, DISC_features=features)
    # 得到训练数据拟合后的test_X
    test_X = data_loader.get_data()
    # 然后将测试数据传递给SVM
    svm_model.load_data(test_X)
    # 然后进行预测
    predictions = svm_model.evaluate()

    if progress_callback:
        progress_callback(90, "Loading data...")

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Time: {:.4f} minutes".format(duration_minutes))
    # 这里获取的就是最终的预测结果,predictions

    combined_predict_list = list(zip(non_b0_indices, predictions))

    return combined_predict_list
