# import tensorflow as tf
# # from tensorflow import keras
# from tensorflow.python import keras
# import os
# import numpy as np
# import nibabel as nib
# import sys
# import json
#
#
#
# def predict_from_saved(a, b):
#     # 定义目标尺寸
#     TARGET_SHAPE = (128, 128, 70)  # (x, y, z)
#
#     # import matplotlib.pyplot as plt
#
#     # Jayse Weaver, weaverjaysem@gmail.com
#     # Evaluating CNN described in https://doi.org/10.1162/imag_a_00023
#
#     # Below example processes two individual diffusion directions
#     # For a 4D NIfTI file, load the entire diffusion volume,
#     # preprocess to [128, 128, 70, # directions], and pass each
#     # diffusion direction volume through the loadModel.predict function
#     # in a loop. Record output prediction to classify each direction volume.
#
#     # load the keras model
#     loadModel = keras.models.load_model('E:\\APP\\Slicer 5.8.1\\QCNet\\QCNet\\Inf_QC\\models\\fold1', compile=False)
#
#     # loadModel = keras.models.load_model(r'')
#
#     threshold = 0.50  # training threshold was 0.50, threshold-moving could be used to
#
#     # path_list = ["example_artifact.nii.gz", "example_normal.nii.gz"]  # path to example nifti volume
#     path_list = ["G:\Artifact_sub\sub-046\dwi.nii.gz"]  # path to example nifti volume
#
#     def pad_or_crop(data, target_size):
#         """自适应填充或裁剪数据到目标尺寸"""
#         current_size = data.shape
#
#         # 计算每个维度的填充/裁剪量
#         start = []
#         end = []
#         pad_width = []
#
#         for dim in range(3):
#             diff = target_size[dim] - current_size[dim]
#
#             if diff > 0:  # 需要填充
#                 pad_before = diff // 2
#                 pad_after = diff - pad_before
#                 pad_width.append((pad_before, pad_after))
#
#                 # 原始数据索引范围
#                 start.append(0)
#                 end.append(current_size[dim])
#             else:  # 需要裁剪
#                 pad_width.append((0, 0))
#
#                 # 计算裁剪范围
#                 crop_start = (-diff) // 2
#                 crop_end = current_size[dim] - (-diff - crop_start)
#                 start.append(crop_start)
#                 end.append(crop_end)
#
#         # 先进行填充
#         padded_data = np.pad(data, pad_width=pad_width[:3], mode='constant', constant_values=0)
#         # 再进行裁剪（如果存在需要裁剪的维度）
#         cropped_data = padded_data[
#                        start[0]:end[0],
#                        start[1]:end[1],
#                        start[2]:end[2]
#                        ]
#
#         return cropped_data
#
#     # for idx, filename in enumerate(path_list):
#     #     print('evaluating: ', filename)
#     #
#     #     # load test case
#     #     niftiVol = nib.load(filename)
#     #
#     #     # create appropriately sized numpy array for evaluation
#     #     X = np.empty((1, 128, 128, 70, 1))  # data must be zero padded or cropped if not [128, 128, 70]
#     #     X[0, :, :, :, 0] = np.array(niftiVol.dataobj)
#     #     X.astype('float32')
#     #
#     #     # evaluate
#     #     pred = loadModel.predict(X)
#     #     print(pred)
#     #
#     #     # assign label
#     #     if pred > threshold:
#     #         print('Motion artifact detected')
#     #     else:
#     #         print('No artifact detected')
#
#     # 修改后的处理循环
#     for idx, filename in enumerate(path_list):
#         print('Evaluating:', filename)
#
#         # 加载数据
#         nifti_vol = nib.load(filename)
#         original_data = np.asarray(nifti_vol.dataobj).astype(np.float32)
#
#         for v in range(original_data.shape[3]):
#             print("volume:", v)
#             # 预处理步骤（根据实际需求添加，例如归一化）
#             processed_data = original_data[..., v] / (np.max(original_data[..., v]))
#
#             # 自适应尺寸调整
#             # adjusted_data = pad_or_crop(original_data[...,v], TARGET_SHAPE)
#             adjusted_data = pad_or_crop(processed_data, TARGET_SHAPE)
#
#             # 添加通道维度和批量维度
#             X = adjusted_data[np.newaxis, ..., np.newaxis]  # 形状变为 (1, 128, 128, 70, 1)
#
#             # 模型预测
#             pred = loadModel.predict(X)
#             print('Prediction score:', pred)
#
#             # 判断结果
#             if pred > threshold:
#                 print('Motion artifact detected')
#             else:
#                 print('No artifact detected')
#
#         """
#         动态尺寸处理：
#         pad_or_crop
#         函数会自动处理三种情况：
#         原始尺寸 > 目标尺寸 → 中心裁剪
#         原始尺寸 < 目标尺寸 → 零填充
#         原始尺寸 = 目标尺寸 → 直接使用
#         保持空间关系：
#         使用中心裁剪 / 对称填充，最大程度保留解剖结构的空间信息
#         填充使用零值（可根据需求修改为其他值）
#         内存优化：
#         使用np.newaxis代替np.empty预分配，避免内存浪费
#         支持任意原始尺寸输入（包括各维度尺寸不一致的情况）
#         扩展性：
#         可修改TARGET_SHAPE适应不同模型需求
#         可添加预处理步骤（如归一化）到注释位置
#
#         """
#
#     return 1, a, b
#
#
# if __name__ == '__main__':
#     # 从命令行参数获取输入（你也可以用 stdin + json）
#     # 参数列表
#     # if len(sys.argv) != 3:
#     #     print(json.dumps({"error": "Usage: test.py a b"}))
#     #     sys.exit(1)
#
#     a = float(sys.argv[1])
#     b = float(sys.argv[2])
#     result, x, y = predict_from_saved(a, b)
#
#     # 输出 JSON 给父进程
#     print(json.dumps({"result": result, "x": x, "y": y}))
