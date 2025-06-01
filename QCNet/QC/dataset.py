import os
import nibabel as nib
import h5py
import numpy as np
import torch
from dipy.io import read_bvals_bvecs
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from dipy.io.image import load_nifti
from os.path import join


def z_score(volume):
    std = np.std(volume)
    mean = np.mean(volume)
    return (volume - mean) / std


class Train_H5Dataset(Dataset):
    def __init__(self, h5_file_path, artifact_type, val_name_list, transform=None):
        self.h5_file_path = h5_file_path
        self.artifact_type = artifact_type
        self.transform = transform
        self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
        self.ground_truth_h5_path = join(h5_file_path, f"good.h5")
        self.mask_h5_path = join(h5_file_path, f"mask.h5")

        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            all_keys = list(h5_file.keys())
            self.dataset_names = [key for key in all_keys if any(sub in key for sub in val_name_list)]

    def __len__(self):
        return len(self.dataset_names)

    def __getitem__(self, idx):
        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            artifact_data = h5_file[dataset_name][:]

        sub = dataset_name.split('_')[0]
        # sub1 = dataset_name.split('_')[1]
        # sub = sub+'_'+sub1
        with h5py.File(self.mask_h5_path, 'r') as h5_file:
            mask = h5_file[sub][:]

        if self.artifact_type =='good':
            label = 0
        else:
            label =1
        # 将数据转换为单通道3D张量
        slice_art_data = torch.tensor(z_score(artifact_data * mask), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # [1]

        return slice_art_data,label


# class Train_H5Dataset(Dataset):
#     def __init__(self, h5_file_path, artifact_type, val_name_list, transform=None):
#         self.h5_file_path = h5_file_path
#         self.artifact_type = artifact_type
#         self.transform = transform
#         self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
#         self.ground_truth_h5_path = join(h5_file_path, f"good.h5")
#         self.mask_h5_path = join(h5_file_path, f"mask.h5")
#         self.artifact_data_list = []
#         self.gt_data_list = []
#         self.mask_data_list = []
#
#         with h5py.File(self.mask_h5_path, 'r') as h5_file:
#             all_keys = list(h5_file.keys())
#             self.dataset_names = [key for key in all_keys if any(sub in key for sub in val_name_list)]
#             for name in self.dataset_names:
#                 self.mask_data_list.append(
#                     np.expand_dims(h5_file[name][...], axis=0))  # f[name][...] 获取该键对应的128*128*128数据
#
#         # self.mask_data_list = np.array(self.mask_data_list)
#         # self.mask_data_list = self.mask_data_list[:, np.newaxis, :, :, :]
#
#         with h5py.File(self.artifact_h5_path, 'r') as h5_file:
#             all_keys = list(h5_file.keys())
#             self.dataset_names = [key for key in all_keys if any(sub in key for sub in val_name_list)]
#             for name in self.dataset_names:
#                 self.artifact_data_list.append(
#                     np.expand_dims(z_score(h5_file[name][...]), axis=0))  # f[name][...] 获取该键对应的128*128*128数据
#
#         # self.artifact_data_list = np.array(self.artifact_data_list)
#         # self.artifact_data_list = self.artifact_data_list[:, np.newaxis, :, :, :]
#         # print(f"{artifact_type}数据准备好")
#
#         with h5py.File(self.ground_truth_h5_path, 'r') as h5_file:
#             all_keys = list(h5_file.keys())
#             self.dataset_names = [key for key in all_keys if any(sub in key for sub in val_name_list)]
#             for name in self.dataset_names:
#                 self.gt_data_list.append(
#                     np.expand_dims(z_score(h5_file[name][...]), axis=0))  # f[name][...] 获取该键对应的128*128*128数据
#
#         # self.gt_data_list = np.array(self.gt_data_list)
#         # self.gt_data_list = self.gt_data_list[:, np.newaxis, :, :, :]
#         # print("ground_truth数据准备好")
#
#         print(f"{artifact_type}数据准备好")
#
#     def __len__(self):
#         return len(self.dataset_names)
#
#     def __getitem__(self, idx):
#         return self.artifact_data_list[idx], self.gt_data_list[idx], self.mask_data_list[idx]

# class Val_H5Dataset(Dataset):
#     def __init__(self, h5_file_path, artifact_type, transform=None):
#         self.h5_file_path = h5_file_path
#         self.artifact_type = artifact_type
#         self.transform = transform
#         self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
#         self.mask_h5_path = join(h5_file_path, f"mask.h5")
#
#         with h5py.File(self.artifact_h5_path, 'r') as h5_file:
#             self.dataset_names = list(h5_file.keys())
#
#     def __len__(self):
#         return len(self.dataset_names)
#
#     def __getitem__(self, idx):
#         with h5py.File(self.artifact_h5_path, 'r') as h5_file:
#             dataset_name = self.dataset_names[idx]
#             # print("dataset_name:", dataset_name)
#             slice_data = h5_file[dataset_name][:]
#
#         sub = dataset_name.split('_')[0]
#         sub1 = dataset_name.split('_')[1]
#         sub = sub+"_"+sub1
#
#         with h5py.File(self.mask_h5_path, 'r') as h5_file:
#             mask = h5_file[sub][:]
#
#         # if self.transform:
#         #     slice_data = self.transform(slice_data)
#
#         # 将数据转换为单通道3D张量
#
#         slice_art_data = torch.tensor(slice_data[0] * mask, dtype=torch.float32).unsqueeze(0)
#         slice_gt_data = torch.tensor(slice_data[1] * mask, dtype=torch.float32).unsqueeze(0)
#         mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
#         return slice_art_data, slice_gt_data, mask


class Val_H5Dataset(Dataset):
    def __init__(self, h5_file_path, artifact_type, val_name_list, transform=None):
        self.h5_file_path = h5_file_path
        self.artifact_type = artifact_type
        self.transform = transform
        self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
        self.ground_truth_h5_path = join(h5_file_path, f"good.h5")
        self.mask_h5_path = join(h5_file_path, f"mask.h5")

        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            all_keys = list(h5_file.keys())
            self.dataset_names = [key for key in all_keys if any(sub in key for sub in val_name_list)]

    def __len__(self):
        return len(self.dataset_names)

    def __getitem__(self, idx):
        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            artifact_data = h5_file[dataset_name][:]

        sub = dataset_name.split('_')[0]
        # sub1 = dataset_name.split('_')[1]
        # sub = sub+'_'+sub1
        with h5py.File(self.mask_h5_path, 'r') as h5_file:
            mask = h5_file[sub][:]

        if self.artifact_type =='good':
            label = 0
        else:
            label =1
        # 将数据转换为单通道3D张量
        slice_art_data = torch.tensor(z_score(artifact_data * mask), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # [1]

        return slice_art_data,label


class Test_H5Dataset(Dataset):
    def __init__(self, h5_file_path, artifact_type, val_name_list, transform=None):
        self.h5_file_path = h5_file_path
        self.artifact_type = artifact_type
        self.transform = transform
        self.artifact_h5_path = join(h5_file_path, f"{artifact_type}.h5")
        self.ground_truth_h5_path = join(h5_file_path, f"good.h5")
        self.mask_h5_path = join(h5_file_path, f"mask.h5")

        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            all_keys = list(h5_file.keys())
            self.dataset_names = [key for key in all_keys if any(sub in key for sub in val_name_list)]

    def __len__(self):
        return len(self.dataset_names)

    def __getitem__(self, idx):
        with h5py.File(self.artifact_h5_path, 'r') as h5_file:
            dataset_name = self.dataset_names[idx]
            artifact_data = h5_file[dataset_name][:]

        sub = dataset_name.split('_')[0]
        # sub1 = dataset_name.split('_')[1]
        # sub = sub+'_'+sub1
        with h5py.File(self.mask_h5_path, 'r') as h5_file:
            mask = h5_file[sub][:]

        if self.artifact_type =='good':
            label = 0
        else:
            label =1
        # 将数据转换为单通道3D张量
        slice_art_data = torch.tensor(z_score(artifact_data * mask), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # [1]

        return slice_art_data,label


if __name__ == '__main__':
    # data_dir = r'E:/NIMG_small_data/Caffine/Train'
    h5_file_path_1 = r'E:/NIMG_small_data/Caffine/Train/caffine_spike.h5'
    h5_file_path_2 = r'E:/NIMG_small_data/Caffine/Train/caffine_no_artifact.h5'
    # create_h5_file(data_dir, h5_file_path)

    caffine_gt_dataset = H5Dataset(h5_file_path_1, transform=None)
    caffine_spike_dataset = H5Dataset(h5_file_path_2, transform=None)

    # total_dataset = caffine_gt_dataset+caffine_spike_dataset
    total_dataset = ConcatDataset([caffine_gt_dataset, caffine_spike_dataset])
    dataloader = DataLoader(total_dataset, batch_size=2, shuffle=True)
    # 示例：遍历数据
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch.size()}")

    h5_file_path = r'E:/NIMG_small_data/HCP_no_artifact.h5'
    HCP_gt_dataset = H5Dataset(h5_file_path, transform=None)
    dataloader = DataLoader(HCP_gt_dataset, batch_size=2, shuffle=True)
    # 示例：遍历数据
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch.size()}")
