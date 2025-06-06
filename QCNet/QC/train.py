7��K����2	��4��<�ew3�6f'r�|��7, [i��&Py~��~s����Ag��n\ ~J>8�`���AP���������;J��W �r1�#�j�njZ� vτ�����B7�1�CX�.�����_���w|I�ɪ���o?@�3�+�z��5�fa�lK��.#�5�m��C)Jqz���U�[�����i�;Z�mD����S<G�vdPP��nI��o�qM��/��	��z�� C��[�R�Y�JyE��p���q8���:D�h�����tK�������T[qDb�����|\��9���%*�[��7|�/���ʝu.Nf{ްZ(�E�N.~s��H�Q��͡6S���AJ�w:َ�O����q:��lmk�ǲ,��RnJ+���m*0��P@��u����Xլ}y�RCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def trainer(train_dataloader, val_dataloader, test_dataloader, epoch_num):
    print("Start training...")

    train_BCE_loss_list = []
    val_BCE_loss_list = []
    test_BCE_loss_list = []

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    for epoch in range(1, epoch_num + 1):
        model.train()
        train_BCE_loss = []

        correct_train = 0
        total_train = 0
        # data表示artifact,target表示ground_truth
        print("Epoch:", epoch)
        for batch_idx, (data,label) in enumerate(tqdm(train_dataloader)):
            data = data.to(device)
            label = label.to(device)

            data[data < 0] = 0
            # 生成的artifact是float64,但是原始ground_truth是float32,这里应该转成一样的
            # data = data.to(torch.float32)

            optimizer.zero_grad()
            y_pred = model(data)
            loss = BCE_loss(y_pred, label)

            loss.backward()
            optimizer.step()
            #
            # loss1 = haarpsi_loss(output, target)
            # loss = loss1 + loss2
            train_BCE_loss.append(loss.item())
            # 准确率计算：对 logits 应用 sigmoid 转换为概率
            y_pred_prob = torch.sigmoid(y_pred)  # 转为概率
            y_pred_class = (y_pred_prob >= 0.5).float()  # 概率按 0.5 阈值转为类别
            correct = (y_pred_class == label).sum().item()
            total = label.size(0)
            correct_train += correct
            total_train += total

        train_acc = correct_train / total_train
        train_acc_list.append(train_acc)
        # print("Epoch : {}  \t Train Loss: {:.4f}".format(epoch,np.mean(train_loss)))

        # print("Validate Epoch")
        model.eval()
        with torch.no_grad():
            val_BCE_loss = []

            correct_val = 0
            total_val = 0
            for batch_idx1, (data1,label1) in enumerate(tqdm(val_dataloader)):
                data1 = data1.to(device)
                label1 = label1.to(device)
                # 转换为float32
                # data1 = data1.to(torch.float32)
                data1[data1 < 0] = 0

                y_pred1 = model(data1)

                loss = BCE_loss(y_pred1, label1)
                val_BCE_loss.append(loss.item())

                # 验证准确率计算
                y_pred1_prob = torch.sigmoid(y_pred1)  # 转为概率
                y_pred1_class = (y_pred1_prob >= 0.5).float()  # 概率转类别
                correct = (y_pred1_class == label1).sum().item()
                total = label1.size(0)
                correct_val += correct # 1 = 1+1
                total_val += total

        val_acc = correct_val / total_val
        val_acc_list.append(val_acc)

        model.eval()
        with torch.no_grad():
            test_BCE_loss = []
            correct_test = 0
            total_test = 0
            for batch_idx1, (data1,label1) in enumerate(tqdm(test_dataloader)):
                data1 = data1.to(device)
                label1 = label1.to(device)
                # 转换为float32
                # data1 = data1.to(torch.float32)

                data1[data1 < 0] = 0
                y_pred1 = model(data1)

                loss = BCE_loss(y_pred1, label1)

                test_BCE_loss.append(loss.item())
                # 验证准确率计算
                y_pred1_prob = torch.sigmoid(y_pred1)  # 转为概率
                y_pred1_class = (y_pred1_prob >= 0.5).float()  # 概率转类别
                correct = (y_pred1_class == label1).sum().item()
                total = label1.size(0) #
                correct_test += correct # 这里的代码有bug
                total_test += total #

        test_acc = correct_test / total_test
        test_acc_list.append(test_acc)
        print(
            "Epoch: {}  \t Train  BEC Loss {:.4f} ACC : {:.4f}  \t Validate BEC Loss {:.4f} ACC : {:.4f} \t Test BCE BEC Loss {:.4f} ACC : {:.4f}".format(
                epoch,
                np.mean(train_BCE_loss),
                train_acc,
                np.mean(val_BCE_loss),
                val_acc,
                np.mean(test_BCE_loss),
                test_acc
            ))
        print(
            "Epoch: {} Train ACC {:.4f} Valid ACC  {:.4f} Test ACC: {:.4f}".format(epoch, train_acc, val_acc, test_acc))


        train_BCE_loss_list.append(np.mean(train_BCE_loss))
        val_BCE_loss_list.append(np.mean(val_BCE_loss))
        test_BCE_loss_list.append(np.mean(test_BCE_loss))

        torch.save(model.state_dict(), join(model_save_dir, f'3D_QCNet_{epoch}.pth'))

        plt.figure()
        plt.plot(train_BCE_loss_list, label="Train BCE Loss")
        plt.plot(val_BCE_loss_list, label="Valid BCE Loss")
        plt.plot(test_BCE_loss_list, label="Test BCE Loss")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("BCE vs Epoch")
        plt.savefig('BCE loss.png')
        plt.show()

        plt.figure()
        plt.plot(train_acc_list, label="Train ACC")
        plt.plot(val_acc_list, label="Valid ACC")
        plt.plot(test_acc_list, label="Test ACC")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("ACC")
        plt.title("3D-QCNet ACC")
        plt.savefig('3D-QCNet_ACC.png')
        plt.show()

    print("Finished train")
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
    # 加载数据集路径
    train_dir_Caffine = '/data/yjh/Journal_DWI_data_h5_no_padding/Train/Caffeine/'

    # 从验证集里面选取15个sub作为验证
    val_dir_Caffine = '/data/yjh/Journal_DWI_data_h5_no_padding/Train/Caffeine/'
    test_dir_Caffine = '/data/yjh/Journal_DWI_data_h5_no_padding/Train/Caffeine/'
    train_name_list =['sub-015', 'sub-018', 'sub-019', 'sub-020', 'sub-022',
                       'sub-025', 'sub-026', 'sub-033', 'sub-036', 'sub-039',
                       'sub-042', 'sub-045', 'sub-050', 'sub-052', 'sub-053',
                       'sub-054', 'sub-057', 'sub-058', 'sub-061', 'sub-062',
                       'sub-067', 'sub-068', 'sub-071', 'sub-076', 'sub-077',
                       'sub-078', 'sub-081', 'sub-082', 'sub-083', 'sub-084']

    # train_name_list = ['sub-015']

    val_name_list =['sub-087', 'sub-092', 'sub-097', 'sub-102', 'sub-103',
                     'sub-104', 'sub-105', 'sub-112', 'sub-122', 'sub-123']
    # val_name_list = ['sub-053']

    test_name_list =['sub-123', 'sub-134', 'sub-140', 'sub-144', 'sub-145',
                      'sub-146', 'sub-147', 'sub-149', 'sub-154', 'sub-159']
    # test_name_list = ['preop_sub-PAT26']

    model_save_dir = './checkpoints'
    # 超参数定义
    learning_rate = 0.0001
    train_batch_size = 8
    val_batch_size = 8

    epoch = 50
    # choose_b1000_num = 5
    # choose_b3000_num = 10

    artifact_type1 = 'good'
    artifact_type2 = 'ghost'
    artifact_type3 = 'spike'
    artifact_type4 = 'swap'
    artifact_type5 = 'motion'
    artifact_type6 = 'eddy'
    artifact_type7 = 'bias'
    #  artifact_type2 = 'swap'

    # train num of subject and validate num of subject
    # train_sub_num = 100
    # val_sub_num = 100
    # 初始化模型

    model = DenseNet3D(num_classes=1)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # print("Using device: ", device)

    device_id = [0, 1]
    # model = Simple3DCNN()
    model = nn.DataParallel(model, device_ids=device_id).cuda()

    # 加载测试数据集,还原ghost失真
    print("****" * 3 + "Loading  training data..." + "****" * 3)
    # 首先加载ghost
    train_set1 = Train_H5Dataset(h5_file_path=train_dir_Caffine,
                                 artifact_type=artifact_type1,
                                 val_name_list=train_name_list)
    # 然后加载spike
    train_set2 = Train_H5Dataset(h5_file_path=train_dir_Caffine,
                                 artifact_type=artifact_type2,
                                 val_name_list=train_name_list)
    # 然后加载noise
    train_set3 = Train_H5Dataset(h5_file_path=train_dir_Caffine,
                                 artifact_type=artifact_type3,
                                 val_name_list=train_name_list)

    # 然后加载swap
    train_set4 = Train_H5Dataset(h5_file_path=train_dir_Caffine,
                                 artifact_type=artifact_type4,
                                 val_name_list=train_name_list)

    # 然后加载motion
    train_set5 = Train_H5Dataset(h5_file_path=train_dir_Caffine,
                                 artifact_type=artifact_type5,
                                 val_name_list=train_name_list)

    # 加载eddy
    train_set6 = Train_H5Dataset(h5_file_path=train_dir_Caffine,
                                 artifact_type=artifact_type6,
                                 val_name_list=train_name_list)

    # 加载good
    train_set7 = Train_H5Dataset(h5_file_path=train_dir_Caffine,
                                 artifact_type=artifact_type7,
                                 val_name_list=train_name_list)

    train_dataset = train_set1 + train_set2 + train_set3 + train_set4 + train_set5 + train_set6 + train_set7
    print("Train data loading finished")
    # 加载验证数据集######################
    val_set1 = Val_H5Dataset(h5_file_path=val_dir_Caffine,
                             artifact_type=artifact_type1,
                             val_name_list=val_name_list)

    val_set2 = Val_H5Dataset(h5_file_path=val_dir_Caffine,
                             artifact_type=artifact_type2,
                             val_name_list=val_name_list)

    val_set3 = Val_H5Dataset(h5_file_path=val_dir_Caffine,
                             artifact_type=artifact_type3,
                             val_name_list=val_name_list)

    val_set4 = Val_H5Dataset(h5_file_path=val_dir_Caffine,
                             artifact_type=artifact_type4,
                             val_name_list=val_name_list)

    val_set5 = Val_H5Dataset(h5_file_path=val_dir_Caffine,
                             artifact_type=artifact_type5,
                             val_name_list=val_name_list)

    val_set6 = Val_H5Dataset(h5_file_path=val_dir_Caffine,
                             artifact_type=artifact_type6,
                             val_name_list=val_name_list)

    val_set7 = Val_H5Dataset(h5_file_path=val_dir_Caffine,
                             artifact_type=artifact_type7,
                             val_name_list=val_name_list)

    val_dataset = val_set1 + val_set2 + val_set3 + val_set4 + val_set5 + val_set6 + val_set7
    print("Validation data loading finished")

    test_set1 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type1,
                               val_name_list=test_name_list)

    test_set2 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type2,
                               val_name_list=test_name_list)

    test_set3 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type3,
                               val_name_list=test_name_list)

    test_set4 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type4,
                               val_name_list=test_name_list)

    test_set5 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type5,
                               val_name_list=test_name_list)

    test_set6 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type6,
                               val_name_list=test_name_list)

    test_set7 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type7,
                               val_name_list=test_name_list)
    test_dataset = test_set1 + test_set2 + test_set3 + test_set4 + test_set5 + test_set6 + test_set7

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=64)
    valid_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=64)
    print("训练数据大小volume数量:", len(train_dataset))
    print("验证数据大小volume数量:", len(val_dataset))
    print("测试集大小volume数量:", len(test_dataset))
    print("****" * 3 + "Finished loading validate data..." + "****" * 3)

    # BCE_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6, 1]).to(device))
    BCE_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1/6]).to(device))  # 大小为 [1] 的张量)
    # l1_loss = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # 训练模型
    trainer(train_loader, valid_loader, test_loader, epoch)

    # 可视化曲线
    # visualize(loss_train, loss_val)
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Time: {:.4f} minutes".format(duration_minutes))
