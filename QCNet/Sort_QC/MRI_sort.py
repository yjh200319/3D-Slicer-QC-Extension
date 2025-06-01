#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:21:28 2023

@author: sun
"""

import os
import shutil
import fnmatch
import torch
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, EnsureChannelFirstd
from MRI_models import MRIEfficientNet
from MRI_transforms import ExtractSlicesd, SwapDimd
import argparse


def dataset_from_folder(data_root_path):
    # #################### new #############################
    dataset = list()
    dataset.append({'img': data_root_path})

    return dataset # 返回的只是路径,并非最终的dataset,而且是字典类型{'img',img_path}


def MRIsort(input_path):  # input就是数据集的名称 output就是输出路径的名称,会有18个小子文件夹
    # 读取单个数据路径
    pred_set = dataset_from_folder(input_path) # 这里应该是创建dataset,根据数据集的路径,也就是输入路径

    # class_file是一个excel文件
    class_file = pd.read_csv("./pretrained_model/labelmap.csv", index_col=0, header=None) # 读取类别csv文件,创立预测的label
    classes = class_file.index.tolist() # 是一个list数据,['FLAIR_cor','FLAIR_tra',...,'artifact_Motion']
    print("classes:", classes)

    num_slices = 3 # 抽取的切片数量,一个超参数
    gap = 2

    transforms = Compose(
        [
            LoadImaged(keys=['img']),
            EnsureChannelFirstd(keys=['img']),
            Resized(keys=["img"], spatial_size=(128, 128, 21)),
            ScaleIntensityd(keys=['img']),
            ExtractSlicesd(keys=['img'], num_slices=num_slices, gap=gap),
            SwapDimd(keys=['img'])
        ]
    )

    pred_ds = Dataset(pred_set, transform=transforms)
    pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda:0")

    # 输入的channel是多少就是他有多少切片,本质还是2D的
    model = MRIEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                            num_classes=len(classes), pretrained=False, dropout_rate=0.2).to(device)

    weights = torch.load("./pretrained_model/best_model.pth")
    model.load_state_dict(weights)

    artifact_index = [9,16,17]

    model.eval()
    with torch.no_grad():
        for pred_batch in pred_loader:
            pred_images = pred_batch["img"].to(device) # 只选这个batch的img,然后将其放到cuda上
            pred_outputs = model(pred_images).argmax(dim=1) #[b,3,128,128,128] -> [batch,3就是上面选择的切片数量,128,128就是图片的切片大小]
            print("pred_outputs", pred_outputs) # 这里输出的是对应的分类类别,例如[1,15]分类到 'T2_tra'
            for i in range(len(pred_outputs)): # 一个批次里面的img分别进行处理,即单个处理
                class_folder = classes[pred_outputs[i]] # 这里相当于直接把类别打印出来了如 'FLAIR_tra'
                src = pred_batch["img"][i].meta["filename_or_obj"] # 选出进行QC的t1w的路径

                conclusion = 'Pass'
                if pred_outputs[i] in artifact_index:
                    conclusion = 'Artifact '
                print("检测如果如下:")
                print("检测图像路径为:",src)
                print("检测的结果为:",class_folder)
                print("最终修正后的结果:",conclusion)

def main(input_file):
    # MRIsort(args.input, args.output)
    MRIsort(input_file)


if __name__ == '__main__':
    # input_file = r'D:\NIMG_data\Caffine\sub-015\t1w.nii.gz' # 这里就输入具体的.nii.gz文件路径, 例如 'D:\NIMG_data\Caffine\sub-015\t1w.nii.gz'
    input_file = r'F:\QC_MRI_dataset\Caffine\sub-020\sub-020_t1w.nii.gz' # 这里就输入具体的.nii.gz文件路径, 例如 'D:\NIMG_data\Caffine\sub-015\t1w.nii.gz'
    # output_file = ''
    main(input_file)
