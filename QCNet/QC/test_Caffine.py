import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from dataset import Train_H5Dataset, Val_H5Dataset, Test_H5Dataset
from model import DenseNet3D
from os.path import join
import os

from read_dwi import DWI_Dataset

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def tester(val_dataloader, test_dataloader):
    model.eval()
    with torch.no_grad():
        correct_val = 0
        total_val = 0
        for batch_idx1, (data1, label1) in enumerate(tqdm(val_dataloader)):
            data1 = data1.to(device)
            label1 = label1.to(device)
            # 转换为float32
            # data1 = data1.to(torch.float32)
            data1[data1 < 0] = 0

            y_pred1 = model(data1)

            # 验证准确率计算
            y_pred1_prob = torch.sigmoid(y_pred1)  # 转为概率
            y_pred1_class = (y_pred1_prob >= 0.5).float()  # 概率转类别
            correct = (y_pred1_class == label1).sum().item()
            total = label1.size(0)
            correct_val += correct  # 1 = 1+1
            total_val += total

    val_acc = correct_val / total_val

    print("Validation Accuracy:", val_acc * 100, '%')
    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for batch_idx1, (data1, label1) in enumerate(tqdm(test_dataloader)):
            data1 = data1.to(device)
            label1 = label1.to(device)
            # 转换为float32
            # data1 = data1.to(torch.float32)

            data1[data1 < 0] = 0
            y_pred1 = model(data1)

            # 验证准确率计算
            y_pred1_prob = torch.sigmoid(y_pred1)  # 转为概率
            y_pred1_class = (y_pred1_prob >= 0.5).float()  # 概率转类别
            correct = (y_pred1_class == label1).sum().item()
            total = label1.size(0)  #
            correct_test += correct  # 这里的代码有bug
            total_test += total  #

    test_acc = correct_test / total_test
    print("Test Accuracy:", test_acc * 100, '%')
    print("Finished test")

    # return val_acc, test_acc
    # 保存模型


def tester_single_sample(val_dataloader, test_dataloader):
    model.eval()
    with torch.no_grad():
        correct_val = 0
        total_val = 0
        for batch_idx1, (data1, label1) in enumerate(tqdm(val_dataloader)):
            data1 = data1.to(device)
            label1 = label1.to(device)
            # 转换为float32
            # data1 = data1.to(torch.float32)
            data1[data1 < 0] = 0

            y_pred1 = model(data1)

            # 验证准确率计算
            y_pred1_prob = torch.sigmoid(y_pred1)  # 转为概率
            y_pred1_class = (y_pred1_prob >= 0.5).float()  # 概率转类别
            correct = (y_pred1_class == label1).sum().item()
            total = label1.size(0)
            correct_val += correct  # 1 = 1+1
            total_val += total

    val_acc = correct_val / total_val

    print("Validation Accuracy:", val_acc * 100, '%')
    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for batch_idx1, (data1, label1) in enumerate(tqdm(test_dataloader)):
            data1 = data1.to(device)
            label1 = label1.to(device)
            # 转换为float32
            # data1 = data1.to(torch.float32)

            data1[data1 < 0] = 0
            y_pred1 = model(data1)

            # 验证准确率计算
            y_pred1_prob = torch.sigmoid(y_pred1)  # 转为概率
            y_pred1_class = (y_pred1_prob >= 0.5).float()  # 概率转类别
            correct = (y_pred1_class == label1).sum().item()
            total = label1.size(0)  #
            correct_test += correct  # 这里的代码有bug
            total_test += total  #

    test_acc = correct_test / total_test
    print("Test Accuracy:", test_acc * 100, '%')
    print("Finished test")

    return val_acc, test_acc
    # 保存模型

def visualize(train_loss_list, val_loss_list):
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Valid Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("HaarPSI Loss")
    plt.title("HaarPSI Loss vs Epoch")
    plt.savefig('Fig_loss.png')
    plt.show()


if __name__ == '__main__':


    model_dir = './checkpoints/3D_QCNet_36.pth'
    # 超参数定义
    learning_rate = 0.0001
    train_batch_size = 8
    val_batch_size = 8

    epoch = 50
    # choose_b1000_num = 5
    # choose_b3000_num = 10


    #  artifact_type2 = 'swap'

    # train num of subject and validate num of subject
    # train_sub_num = 100
    # val_sub_num = 100
    # 初始化模型
    device_id = [0]
    model = DenseNet3D(num_classes=1)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    model = nn.DataParallel(model, device_ids=device_id).cuda()
    checkpoint = torch.load(model_dir, map_location=device)
    model.load_state_dict(checkpoint)

    # 加载测试数据集,还原ghost失真
    print("****" * 3 + "Loading  Testing data..." + "****" * 3)
    # 首先加载ghost

    # 定义dwi_path, mask_path, bval_path, bvec_path
    dwi_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi.nii.gz'
    mask_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi_mask.nii.gz'
    bval_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi.bval'
    bvec_path = r'E:\Python\pythonProject7\3D-QCNet\sub-046\dwi.bvec'

    # 创建DWI数据集实例
    test_set7 = DWI_Dataset(dwi_path=dwi_path, mask_path=mask_path, bval_path=bval_path, bvec_path=bvec_path)



    # test_dataset = test_set1 + test_set2 + test_set3 + test_set4 + test_set5 + test_set6 + test_set7
    test_dataset = test_set7
  #  valid_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=64)
  #  print("验证数据大小volume数量:", len(val_dataset))
    print("测试集大小volume数量:", len(test_dataset))
    print("****" * 3 + "Finished loading validate data..." + "****" * 3)

    # BCE_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6, 1]).to(device))
    BCE_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1 / 6]).to(device))  # 大小为 [1] 的张量)
    # l1_loss = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # 训练模型
  #  tester(valid_loader, test_loader)

    # 可视化曲线
    # visualize(loss_train, loss_val)
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Time: {:.4f} minutes".format(duration_minutes))

    # 按照artifact的类型,一个一个的测试,然后按照列表的sub名称,求出均值,然后再算方差。

    artifact_type = 'bias'

    caffine_acc_list =[]
    aomic_acc_list =[]
    for name1,name2 in zip(test_name_Caffine_list,test_name_AOMIC_list):
        test_Caffine_set = Val_H5Dataset(h5_file_path=test_dir_Caffine,
                                         artifact_type=artifact_type,
                                         val_name_list=[name1])
        print("Length of test_Caffine set: ", len(test_Caffine_set))
        test_Caffine_dataloader = DataLoader(test_Caffine_set, batch_size=val_batch_size, shuffle=False, num_workers=64)

        test_AOMIC_set = Test_H5Dataset(h5_file_path=test_dir_AOMIC,
                                        artifact_type=artifact_type,
                                        val_name_list=[name2])
        print("Length of test_AOMIC set: ", len(test_AOMIC_set))
        test_AOMIC_dataloader = DataLoader(test_AOMIC_set, batch_size=val_batch_size, shuffle=False, num_workers=64)

        caffine_acc,aomic_acc = tester_single_sample(test_Caffine_dataloader,test_AOMIC_dataloader)
        caffine_acc_list.append(caffine_acc)
        aomic_acc_list.append(aomic_acc)
        print("Name:",name1)
        print("Caffine Accuracy:",caffine_acc*100,'%')
        print("Name:", name2)
        print("AOMIC Accuracy:", aomic_acc * 100, '%')

    print("Artifact-type:", artifact_type)
    print("Mean Caffine ACC:",np.mean(caffine_acc_list)*100,'%')
    print("Std Caffine ACC:",np.std(caffine_acc_list)*100,'%')
    print("Mean AOMIC ACC:",np.mean(aomic_acc_list)*100,'%')
    print("Std AOMIC ACC:",np.std(aomic_acc_list)*100,'%')