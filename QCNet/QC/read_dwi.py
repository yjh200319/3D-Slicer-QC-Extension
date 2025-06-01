from os.path import join

import numpy as np
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti
from torch.utils.data import Dataset


def z_score(volume):
    std = np.std(volume)
    mean = np.mean(volume)
    return (volume - mean) / std


# # 写一个dataset.py文件,根据dwi文件的路径,提供最终的信息
class DWI_Dataset(Dataset):
    def __init__(self, dwi_path, mask_path, bval_path, bvec_path):
        self.dwi_path = dwi_path
        self.mask_path = mask_path
        self.bval_path = bval_path
        self.bvec_path = bvec_path

        # 读取数据,affine一般用不到,除了保存dwi.nii.gz文件之外
        dwi_data, affine = load_nifti(dwi_path, return_img=False)
        mask_data, _ = load_nifti(mask_path, return_img=False)
        # 这里一般只有bvals能用上,它是一个一维数组,例如[0.00,0.001,0.002,1000.01,999.99,1000,3000,2999]
        # 这个长度和dwi文件大小[x,y,z,v]中的v的大小一样的,是一一对应的关系
        # 但是需要进行预处理,因为有些比如0.001或者2999这些采集参数实际上就是对应四舍五入,但有一些误差
        # 所以需要把数据取整一下 [0.00,0.001,0.002,1000.01,999.99,1000,3000,2999] -> [0,0,0,1000,1000,1000,3000,3000]
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        bvals = np.round(bvals / 100) * 100  # 预处理后的bvals

        # 然后获取b!=0的v的索引
        non_b0_indices = np.where(bvals != 0)[0]

        # 然后单个volume归一化(z-score),然后用列表读取
        dwi_volume_list = []
        dwi_volume_name_list = []
        for v in non_b0_indices:
            volume = dwi_data[:, :, :, v]
            volume = z_score(volume * mask_data)  # 只提取大脑区域
            volume = np.expand_dims(volume, axis=0)  # 单通道[x,y,z]->[1,x,y,z]
            dwi_volume_list.append(volume)
            dwi_volume_name_list.append(v)
        # 然后最后变成[length(non_b0_indices),1,x,y,z]的列表
        self.dwi_volume_list = dwi_volume_list
        self.dwi_volume_name_list =dwi_volume_name_list
    def __len__(self):
        return len(self.dwi_volume_list)

    def __getitem__(self, idx):
        # 将数据读取,然后将dwi_data[x,y,z,v]中的b=0的体积去掉形成 [x,y,z,v1],然后再变成[v1,1,x,y,z]的单通道数据
        # v1表示有多少个这样大小为x*y*z的3D体积数据
        return self.dwi_volume_list[idx], self.dwi_volume_name_list[idx]


if __name__ == '__main__':
    dwi_path = r'D:\NIMG_data\Caffine\sub-015\dwi.nii.gz'
    mask_path = r'D:\NIMG_data\Caffine\sub-015\dwi_mask.nii.gz'
    bval_path = r'D:\NIMG_data\Caffine\sub-015\dwi.bval'
    bvec_path = r'D:\NIMG_data\Caffine\sub-015\dwi.bvec'

    dataset = DWI_Dataset(dwi_path=dwi_path, mask_path=mask_path, bval_path=bval_path, bvec_path=bvec_path)
    print("dataset的大小为:", len(dataset))
    # 应该是128,因为原来的dwi.nii.gz大小是[112,112,50,136],里面b=0的有8个3D数据,去掉8个,就是128个3D数据
