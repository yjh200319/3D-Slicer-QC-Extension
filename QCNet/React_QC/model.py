from collections import OrderedDict

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 判断设备，确保 CUDA 可用时使用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpu = torch.cuda.device_count()
print("当前可用GPU数量:", num_gpu)


# 这里加上Unet的定义
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Simple3D_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Simple3D_UNet, self).__init__()
        self.encoder1 = ConvBlock(in_channels, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        # self.encoder4 = ConvBlock(128, 256)
        # self.encoder5 = ConvBlock(256, 512)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # self.upconv4 = UpConvBlock(512, 256)
        # self.upconv3 = UpConvBlock(256, 128)
        self.upconv2 = UpConvBlock(128, 64)
        self.upconv1 = UpConvBlock(64, 32)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        # e4 = self.encoder4(self.pool(e3))
        # e5 = self.encoder5(self.pool(e4))

        # d4 = self.upconv4(e5, e4)
        # d3 = self.upconv3(e4, e3)
        d2 = self.upconv2(e3, e2)
        d1 = self.upconv1(d2, e1)

        out = self.final_conv(d1)
        return out


# ###########SVM模型########################
class SVM_DataLoader:
    def __init__(self, train_paths, DISC_features):
        """
        初始化数据加载器
        :param train_paths: 训练集文件路径列表
        :param val_paths: 验证集文件路径列表
        """
        self.train_paths = train_paths
        self.train_features = None
        self.train_labels = None
        self.val_features = None
        self.val_labels = None
        self.scaler = StandardScaler()  # 统一使用同一个Scaler
        self.DISC_features = DISC_features
        # self.choose_sample = choose_sample

    def load_data(self, paths):
        """
        从路径加载数据（假设数据存储为.npy格式字典）
        :param paths: 数据文件路径列表
        :return: 特征和标签
        """
        features_list = []
        labels_list = []

        for path in paths:
            data = np.load(path, allow_pickle=True).item()
            features_list.append(data['features'])
            labels_list.append(data['labels'])

        # 合并所有路径的数据
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        return features, labels

    def load_data_test(self, DISC_features):
        """
        从路径加载数据（假设数据存储为.npy格式字典）
        :param paths: 数据文件路径列表
        :return: 特征和标签
        """
        features = np.vstack(DISC_features)
        return features

    def load_data_test_single(self, paths, sub_filter=None):
        """
        从路径加载数据，并根据 sub_filter 过滤样本（假设数据存储为.npy格式字典）
        :param paths: 数据文件路径列表
        :param sub_filter: 如果不为None，则只加载名字中包含该子串的样本；
                           如果为None，则按照 self.choose_sample 中的任一子串进行筛选。
        :return: 特征和标签
        """
        features_list = []
        labels_list = []

        for path in paths:
            data = np.load(path, allow_pickle=True).item()
            if sub_filter is not None:
                # 只选择名字中包含 sub_filter 的样本
                indices = [i for i, name in enumerate(data['names']) if sub_filter in name]
            else:
                # 默认：选择名字中包含 self.choose_sample 中任一子串的样本
                indices = [i for i, name in enumerate(data['names'])
                           if any(sub in name for sub in self.choose_sample)]
            filtered_features = data['features'][indices]
            filtered_labels = np.array(data['labels'])[indices]
            features_list.append(filtered_features)
            labels_list.append(filtered_labels)

        # 合并所有路径的数据
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        return features, labels

    def preprocess_data(self, features, is_train=True):
        """
        进行数据预处理：包括反转 HaarPSI 和 标准化所有特征
        :param features: 输入的特征数据
        :return: 预处理后的特征
        """
        # 提取 MSE, LPIPS 和 HaarPSI
        MSE = features[:, 0]
        LPIPS = features[:, 1]
        HaarPSI = features[:, 2]

        # 反转 HaarPSI（因为它是越大越好）
        HaarPSI = 1 - HaarPSI  # 这里使用 1 - HaarPSI
        # 重新组合特征
        features = np.column_stack([MSE, LPIPS, HaarPSI])

        if is_train:
            features = self.scaler.fit_transform(features)
        else:
            # 如果是测试数据，使用训练集计算的均值和方差进行transform

            features = self.scaler.transform(features)

        return features

    def get_data(self):
        """
        加载并预处理训练集和验证集数据
        :return: 训练集和验证集数据
        """
        # 加载训练集和验证集数据
        self.train_features, self.train_labels = self.load_data(self.train_paths)
        self.val_features = self.load_data_test(self.DISC_features)

        # 预处理训练集和验证集特征
        self.train_features = self.preprocess_data(self.train_features, is_train=True)
        self.val_features = self.preprocess_data(self.val_features, is_train=False)

        return self.val_features


class SVMModel:
    def __init__(self, C=1.0, kernel='linear', max_iter=1000, random_state=42):
        """
        初始化SVM模型
        :param C: SVM的正则化参数，控制模型的复杂度
        :param kernel: 核函数类型，可以是 'linear', 'rbf', 'poly' 等
        :param random_state: 随机种子，用于保证结果可复现
        """
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None

    # def load_data(self, train_features, train_labels, val_features, val_labels):
    def load_data(self, val_features):
        """
        加载训练数据和验证数据
        :param train_features: 训练集特征
        :param train_labels: 训练集标签
        :param val_features: 验证集特征
        :param val_labels: 验证集标签
        """
        # self.X_train = train_features
        # self.y_train = train_labels
        # self.y_val = val_labels

        self.X_val = val_features

    def train(self):
        """
        训练SVM模型
        """
        """
        self.model = SVC(C=self.C, kernel=self.kernel, class_weight={0: 124, 1: 1}, random_state=self.random_state,
                         max_iter=self.max_iter) 这个得到的0无伪影的recall较高,同时1的recall也比较高,有95%,缺点是0的precision比较低
        """

        self.model = SVC(C=self.C, kernel=self.kernel, class_weight={0: 99, 1: 1}, random_state=self.random_state,
                         max_iter=self.max_iter)
        self.model.fit(self.X_train, self.y_train)
        print("SVM model trained successfully.")

    def evaluate(self):
        """
        在验证集上评估模型表现
        """
        y_pred = self.model.predict(self.X_val)
        # accuracy = accuracy_score(self.y_val, y_pred)
        # print("y_val: ", self.y_val)
        # print("y_pred: ", y_pred)
        # print(f"Accuracy on validation set: {accuracy:.4f}")
        # print("Classification Report:")
        print("SVM model evaluated successfully:")
        print("预测的标签如下:", y_pred)
        return y_pred

    def save_model(self, filename='svm_model.pkl'):
        """
        保存训练好的模型到文件
        :param filename: 保存的模型文件路径
        """
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='svm_model.pkl'):
        """
        从文件加载训练好的模型
        :param filename: 模型文件路径
        """
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")

    def evaluate_with_std(self, data_loader):
        """
        针对每个 sub_name 计算对应样本的正确率，并返回平均值和标准差
        :param data_loader: DataLoader 实例，其中包含 choose_sample
        :return: (平均正确率, 标准差)
        """
        acc_sub_list = []

        # 针对每个子串单独加载数据进行评估
        for sub_name in data_loader.choose_sample:
            print("sub_name", sub_name)
            # 加载当前子串对应的验证数据
            val_features, val_labels = data_loader.load_data_test_single(data_loader.val_paths, sub_name)

            # 若加载到的数据为空则跳过（可根据实际情况处理）
            if val_features.size == 0:
                print(f"Warning: 没有找到 {sub_name} 对应的样本。")
                continue

            # 对加载到的数据进行预处理（注意：此处使用训练时fit好的 scaler）
            val_features = data_loader.preprocess_data(val_features, is_train=False)

            # 预测并计算正确率
            y_pred = self.model.predict(val_features)
            acc = accuracy_score(val_labels, y_pred)
            acc_sub_list.append(acc)
            print(f"Accuracy for {sub_name}: {acc:.4f}")

        if len(acc_sub_list) == 0:
            print("没有找到任何子样本数据用于评估。")
            return None, None

        # 计算均值和标准差
        mean_acc = np.mean(acc_sub_list)
        std_acc = np.std(acc_sub_list)
        print(f"Mean Accuracy: {mean_acc * 100:.2f}%, Standard Deviation: {std_acc * 100:.2f}%")
        return mean_acc, std_acc


class QualityAssessmentPipeline:
    def __init__(self, model_path):
        self.unet = self.load_unet(model_path['unet'])  # 加载3D-UNet
        self.svm = self.load_svm(model_path['svm'])  # 加载svm

    def load_unet(self, model_path):
        model = Simple3D_UNet(in_channels=1, out_channels=1)
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

    def load_svm(self, model_path):
        svm_model = SVMModel(C=0.4, max_iter=600, kernel='linear')  # 注意，此处参数可根据实际情况传入
        svm_model.load_model(filename=model_path)

        return svm_model

