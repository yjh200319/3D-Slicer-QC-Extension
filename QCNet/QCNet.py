import os
import tempfile
from typing import Optional

import numpy as np
import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from dipy.io.image import load_nifti, save_nifti

from slicer import vtkMRMLScalarVolumeNode

from test_single_sub import test_single, test_single_by_react_qc
from dipy.io import read_bvals_bvecs
import subprocess
import json

current_dir = os.path.dirname(os.path.abspath(__file__))


#
# QCNet
#

## 用于信息初始化
class QCNet(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("QCNet")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#QCNet">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # QCNet1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="QCNet",
        sampleName="QCNet1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "QCNet1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="QCNet1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="QCNet1",
    )

    # QCNet2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="QCNet",
        sampleName="QCNet2",
        thumbnailFileName=os.path.join(iconsPath, "QCNet2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="QCNet2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="QCNet2",
    )


# 以上是插件自带的，不用管


#
# QCNetParameterNode
#


## 模块所需要的参数,
# InputVolume-要设置的阈值
# imageThreshold-设置输入阈值的值
#
@parameterNodeWrapper
class QCNetParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode


#
# QCNetWidget
#


"""
    主要的窗口界面显示。
    可以完成各种界面设置和设计以及连接QCNetLogic
    可以https://github.com/Slicer..
"""


class QCNetWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/QCNet.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = QCNetLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.DMRIApplyButton.connect("clicked(bool)", self.DMRIOnApplyButton)
        self.ui.MRIApplyButton.connect("clicked(bool)", self.MRIOnApplyButton)
        # 设置saveDiscard的button,然后执行点击后接口逻辑
        self.ui.pushButton_save_discard.connect("clicked(bool)", self.onSaveDiscardButton)

        self.PathLineEdit_mask_dMRI = self.ui.PathLineEdit_mask_dMRI
        self.PathLineEdit_mask_MRI = self.ui.PathLineEdit_mask_MRI

        # self.PathLineEdit_mask = self.ui.PathLineEdit_mask
        self.PathLineEdit_bval = self.ui.PathLineEdit_bval
        self.PathLineEdit_bvec = self.ui.PathLineEdit_bvec
        # self.PathLineEdit_model = self.ui.PathLineEdit_model

        # 添加一个新的进度条
        self.ui.DMRIProgressBar.setVisible(False)
        self.ui.DMRIProgressBar.setValue(0)

        self.ui.MRIProgressBar.setVisible(False)
        self.ui.MRIProgressBar.setValue(0)
        # Make sure parameter node is initialized (needed for module reload)

        # 添加复选框
        # 初始化方法选择器
        self.ui.DMRIMethodSelector.addItems(["3D-QCNet", "React-QC", "Inf-QC"])
        self.ui.DMRIMethodSelector.setCurrentIndex(0)  # 默认选中第一个

        # self.ui.MRIMethodSelector.addItems(["Light-Artifact-QC", "Stochastic-MC-QC", "MRI-QC-3"])
        self.ui.MRIMethodSelector.addItems(["Stochastic-MC-QC"])
        self.ui.MRIMethodSelector.setCurrentIndex(0)  # 默认选中第一个

        # 初始化消息框
        self.initializeMessageBox()

        self.initializeParameterNode()

    def appendMessage(self, text, msg_type="info"):
        """通用方法：添加带样式的消息"""
        color_map = {
            "info": "black",
            "success": "#008000",
            "error": "#FF0000"
        }
        html = f'<span style="color:{color_map[msg_type]}">{text}</span>'
        self.ui.LogTextEdit_MRI.append(html)  # 使用 HTML 格式化文本
        self.ui.LogTextEdit_MRI.verticalScrollBar().setValue(
            self.ui.LogTextEdit_MRI.verticalScrollBar().maximum
        )  # 自动滚动到底部

    def initializeMessageBox(self):
        """"初始化消息框"""
        self.ui.LogTextEdit_MRI.clear()
        self.ui.LogTextEdit_MRI.setReadOnly(True)
        self.ui.LogTextEdit_MRI.append(">>> 系统已就绪")

        # 可以选设置字体样式
        self.ui.LogTextEdit_MRI.setStyleSheet(
            "QLineEdit { background-color: #F0F0F0; font-family: Consolas; }"
        )

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[QCNetParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.DMRIApplyButton.toolTip = _("Compute output volume")
            self.ui.DMRIApplyButton.enabled = True

            self.ui.MRIApplyButton.toolTip = _("Compute output volume")
            self.ui.MRIApplyButton.enabled = True
        else:
            self.ui.DMRIApplyButton.toolTip = _("Select input and output volume nodes")
            self.ui.DMRIApplyButton.enabled = True

            self.ui.MRIApplyButton.toolTip = _("Select input and output volume nodes")
            self.ui.MRIApplyButton.enabled = True

    def DMRIOnApplyButton(self) -> None:
        ## 添加页面的按钮,交互式的
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # 显示进度条
            selected_method = self.ui.DMRIMethodSelector.currentText
            try:
                # 创建进度条更新时间
                def update_progress(progress, message=None):
                    self.ui.DMRIProgressBar.setValue(progress)
                    if message:
                        slicer.util.showStatusMessage(message, 2000)
                    slicer.app.processEvents()

                self.ui.DMRIProgressBar.setVisible(True)
                self.ui.DMRIProgressBar.setValue(0)
                slicer.app.processEvents()

                mask = self.PathLineEdit_mask_dMRI.currentPath
                bval = self.PathLineEdit_bval.currentPath
                bvec = self.PathLineEdit_bvec.currentPath
                # model = self.PathLineEdit_model.currentPath
                # Compute output
                self.logic.process(self.ui.DMRIInputSelector.currentNode(),
                                   mask, bval, bvec,
                                   progress_callback=update_progress,
                                   method=selected_method  # 传递回调
                                   )
            finally:
                # 完成隐藏进度条
                self.ui.DMRIProgressBar.setVisible(False)

    def MRIOnApplyButton(self) -> None:
        ## 添加页面的按钮,交互式的
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # 显示进度条
            selected_method = self.ui.MRIMethodSelector.currentText
            try:
                # 创建进度条更新时间
                def update_progress(progress, message=None):
                    self.ui.MRIProgressBar.setValue(progress)
                    if message:
                        slicer.util.showStatusMessage(message, 2000)
                    slicer.app.processEvents()

                self.ui.MRIProgressBar.setVisible(True)
                self.ui.MRIProgressBar.setValue(0)
                slicer.app.processEvents()

                mask = self.PathLineEdit_mask_MRI.currentPath
                print("mask", mask)
                # model = self.PathLineEdit_model.currentPath
                # Compute output
                QC_Type, QC_Prob = self.logic.process_MRI(self.ui.MRIInputSelector.currentNode(),
                                                          mask,
                                                          progress_callback=update_progress,
                                                          method=selected_method  # 传递回调
                                                          )

                self.appendMessage("处理完成！", "success")
                # self.appendMessage("结果如下:", "success")
                self.appendMessage(f"判断结果: {QC_Type}, 置信度: {QC_Prob}%", "success")

            finally:
                # 完成隐藏进度条
                self.ui.MRIProgressBar.setVisible(False)

    # 这里定义点击saveDiscard的按钮执行逻辑
    def onSaveDiscardButton(self) -> None:
        ## 添加页面的按钮,交互式的
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # 这里获取要保存的path,例如当前目录下,去掉最后的子目录
            # 这里我需要去读取dwi.nii.gz文件
            # 以及bval和bvec
            # 然后根据QC的结果,去有选择的去保留其中的索引,最终点击save以后
            # 将discard的数据保存到指定的目录下面

            # 这里读取原始的dwi文件的路径
            origin_dwi_path = self.ui.DMRIInputSelector.currentNode().GetStorageNode().GetFullNameFromFileName()
            mask_path = self.PathLineEdit_mask_dMRI.currentPath
            bval_path = self.PathLineEdit_bval.currentPath
            bvec_path = self.PathLineEdit_bvec.currentPath

            # 然后这里读取然后保存

            save_path = os.path.dirname(mask_path)
            print("origin_path", origin_dwi_path)
            print("mask_path", mask_path)
            print("bval_path", bval_path)
            print("bvec_path", bvec_path)
            print("save_path", save_path)

            # Compute output
            self.logic.process_saveDiscard(origin_dwi_path, bval_path, bvec_path, save_path)


#
# QCNetLogic
#


"""
    此类应试下内所有实际逻辑处理和模块完成的计算。
    接口应该使得其他python代码可以导入这个类并使用该功能,
    而不需要Widget的实例,就是没有界面的纯逻辑代码,
    继承自ScriptedLoadableModuleLogic基类
"""


class QCNetLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return QCNetParameterNode(super().getParameterNode())

    def _process_3d_qcnet(self,
                          inputVolume,
                          mask,
                          bval,
                          bvec,
                          progress_callback=None):

        import time
        import logging
        startTime = time.time()
        logging.info("Processing started")

        # model = r'E:\APP\Slicer 5.8.1\QCNet\QCNet\QC\3D_QCNet_model.pth'
        model = os.path.join(current_dir, "QC/3D_QCNet_model.pth")
        # 分阶段的进度条
        if progress_callback:
            progress_callback(10, "Loading data...")

        inputFilePath = inputVolume.GetStorageNode().GetFullNameFromFileName()

        orig_name = inputFilePath

        print("OKK")

        volume_classify_results_list = test_single(
            orig_name,
            mask, bval, bvec, model,
            test_batch_size=2,
            # progress_callback = progress_callback
            progress_callback=progress_callback
        )
        print("volume_classify_results_list")
        print(volume_classify_results_list)
        # 只保留label=1的索引列表,表示要drop掉这些
        self.discard_index_list = [index for index, label in volume_classify_results_list if label == 1]

        # 调用函数显示表格
        self.create_table_qcnet(volume_classify_results_list)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")

        if progress_callback:
            progress_callback(100, "Done!")

        # ##############################################
        # # 主进程
        # # 构建调用命令（Linux / Mac）
        # # python_executable = "QCNet/Inf_QC/venv/Scripts/python"
        # python_executable = os.path.join(current_dir, "Inf_QC/venv/Scripts/python")
        # # test_script = "QCNet/Inf_QC/predict_from_saved_1.py"
        # test_script = os.path.join(current_dir, "Inf_QC/predict_from_saved_1.py")
        #
        # # Windows 下路径示例：
        # # python_executable = "test_env\\venv\\Scripts\\python.exe"
        # # test_script = "test_env\\test.py"
        #
        # # 输入参数
        # a, b = 3, 7
        #
        # # 调用子进程并捕获输出
        # result = subprocess.run(
        #     [python_executable, test_script, str(a), str(b)],
        #     # capture_output=True,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT,
        #     text=True,
        # )
        #
        # # 分析结果
        # try:
        #     output = json.loads(result.stdout)
        #     print("output:")
        #     print(output)
        #     if "result" in output:
        #         print(f"Result from test.py: {output['result']}, {output['x']}, {output['y']}")
        #     else:
        #         print(f"Error from test.py: {output.get('error')}")
        # except json.JSONDecodeError:
        #     print("Invalid output from test.py:")
        #     print(result.stdout)

        # #############################################

    def create_table_qcnet(self, volume_classify_results_list):
        # 创建一个 vtkTable 对象来存储表格数据
        table = vtk.vtkTable()

        # 创建列 (Volume idx 和 Class)
        volume_idx_column = vtk.vtkStringArray()
        volume_idx_column.SetName("Volume idx")
        table.AddColumn(volume_idx_column)

        class_column = vtk.vtkStringArray()
        class_column.SetName("Class")
        table.AddColumn(class_column)

        # 将数据填充到表格
        for volume_idx, vol_class in volume_classify_results_list:
            row = table.InsertNextBlankRow()  # 插入新的一行
            table.SetValue(row, 0, str(volume_idx))  # Volume idx 列
            if vol_class == 0:
                table.SetValue(row, 1, str("Pass"))  # Class 列
            elif vol_class == 1:
                table.SetValue(row, 1, str("Failed"))  # Class 列
        # 创建并显示节点
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "3D-QCNet Results")
        table_node.SetAndObserveTable(table)

        # 为 vtkMRMLTableNode 设置列名和数据
        for col_idx in range(table.GetNumberOfColumns()):
            column = table.GetColumn(col_idx)
            table_node.AddColumn(column)  # 将列添加到 vtkMRMLTableNode

        # 刷新 Slicer 界面
        slicer.app.processEvents()

    # ##### React-QC的方法
    def _process_react_qc(self, inputVolume, mask, bval, bvec, progress_callback=None):
        # 新增 React-QC 处理逻辑
        inputFilePath = inputVolume.GetStorageNode().GetFullNameFromFileName()

        dwi_path = inputFilePath

        # 分阶段的进度条
        if progress_callback:
            progress_callback(10, "Loading data...")

        react_classify_results = test_single_by_react_qc(dwi_path, mask, bval, bvec,
                                                         progress_callback=progress_callback)
        print("react-QC:")
        print(react_classify_results)

        # 只保留label=1的索引列表,表示要drop掉这些
        self.discard_index_list = [index for index, label in react_classify_results if label == 1]

        self.create_table_react_qc(react_classify_results)

    def create_table_react_qc(self, volume_classify_results_list):
        # 创建一个 vtkTable 对象来存储表格数据
        table = vtk.vtkTable()

        # 创建列 (Volume idx 和 Class)
        volume_idx_column = vtk.vtkStringArray()
        volume_idx_column.SetName("Volume idx")
        table.AddColumn(volume_idx_column)

        class_column = vtk.vtkStringArray()
        class_column.SetName("Class")
        table.AddColumn(class_column)

        # 将数据填充到表格
        for volume_idx, vol_class in volume_classify_results_list:
            row = table.InsertNextBlankRow()  # 插入新的一行
            table.SetValue(row, 0, str(volume_idx))  # Volume idx 列
            if vol_class == 0:
                table.SetValue(row, 1, str("Pass"))  # Class 列
            elif vol_class == 1:
                table.SetValue(row, 1, str("Failed"))  # Class 列
        # 创建并显示节点
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "React-QC Results")
        table_node.SetAndObserveTable(table)

        # 为 vtkMRMLTableNode 设置列名和数据
        for col_idx in range(table.GetNumberOfColumns()):
            column = table.GetColumn(col_idx)
            table_node.AddColumn(column)  # 将列添加到 vtkMRMLTableNode

        # 刷新 Slicer 界面
        slicer.app.processEvents()

    # ###########Inf_QC #########################
    def _process_inf_qc(self, inputVolume, mask, bval, bvec, progress_callback=None):

        # 新增 React-QC 处理逻辑
        inputFilePath = inputVolume.GetStorageNode().GetFullNameFromFileName()

        dwi_path = inputFilePath

        # # 分阶段的进度条
        # if progress_callback:
        #     progress_callback(10, "Loading data...")
        #
        # react_classify_results = test_single_by_react_qc(dwi_path, mask, bval, bvec,
        #                                                  progress_callback=progress_callback)
        # print("react-QC:")
        # print(react_classify_results)
        #
        # # 只保留label=1的索引列表,表示要drop掉这些
        # self.discard_index_list = [index for index, label in react_classify_results if label == 1]
        #
        # self.create_table_react_qc(react_classify_results)

        # ###############调用子线程
        # 主进程
        # 构建调用命令（Linux / Mac）
        # python_executable = "QCNet/Inf_QC/venv/Scripts/python"
        python_executable = os.path.join(current_dir, "Inf_QC/venv/Scripts/python")
        # test_script = "QCNet/Inf_QC/predict_from_saved_1.py"
        test_script = os.path.join(current_dir, "Inf_QC/predict_from_saved.py")

        # ########加载图像
        # Windows 下路径示例：
        # python_executable = "test_env\\venv\\Scripts\\python.exe"
        # test_script = "test_env\\test.py"

        # 输入参数
        # a, b = 3, 7

        dwi, affine = load_nifti(dwi_path, return_img=False)
        mask_data, _ = load_nifti(mask, return_img=False)
        bvals_data, _ = read_bvals_bvecs(bval, bvec)

        # 创建临时文件并保存数组
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as dwi_file, \
                tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as mask_file, \
                tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as bval_file:

            np.save(dwi_file.name, dwi)
            np.save(mask_file.name, mask_data)
            np.save(bval_file.name, bvals_data)

            # 获取临时文件路径
            dwi_path = dwi_file.name
            mask_path = mask_file.name
            bval_path = bval_file.name

        # 调用子进程
        result = subprocess.run(
            [python_executable, test_script, dwi_path, mask_path, bval_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # 将 stderr 与 stdout 分离
            # stderr=subprocess.STDOUT,
            text=True,
            # encoding='utf-8'  # 确保编码一致
        )

        # 清理临时文件
        os.unlink(dwi_path)
        os.unlink(mask_path)
        os.unlink(bval_path)

        # # 调用子进程并捕获输出
        # result = subprocess.run(
        #     [python_executable, test_script, str(a), str(b),str()],
        #     # capture_output=True,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.STDOUT,
        #     text=True,
        # )

        # 分析结果
        inf_classify_results = []
        try:
            output = json.loads(result.stdout)
            print("output:")
            print(output)
            if "result" in output:
                print(f"Result from test.py: {output['result']}")
                # inf_classify_results = {output['result']}
                inf_classify_results = output['result']

            else:
                print(f"Error from test.py: {output.get('error')}")
        except json.JSONDecodeError:
            print("Invalid output from test.py:")
            print(result.stdout)

        print("inf_classify_results:")
        print(inf_classify_results)
        # 只保留label=1的索引列表,表示要drop掉这些
        self.discard_index_list = [index for index, label in inf_classify_results if label == 1]
        # #####建立label
        self.create_table_inf_qc(inf_classify_results)
        # ######################

    def create_table_inf_qc(self, volume_classify_results_list):
        # 创建一个 vtkTable 对象来存储表格数据
        table = vtk.vtkTable()

        # 创建列 (Volume idx 和 Class)
        volume_idx_column = vtk.vtkStringArray()
        volume_idx_column.SetName("Volume idx")
        table.AddColumn(volume_idx_column)

        class_column = vtk.vtkStringArray()
        class_column.SetName("Class")
        table.AddColumn(class_column)

        # 将数据填充到表格
        for volume_idx, vol_class in volume_classify_results_list:
            row = table.InsertNextBlankRow()  # 插入新的一行
            table.SetValue(row, 0, str(volume_idx))  # Volume idx 列
            if vol_class == 0:
                table.SetValue(row, 1, str("Pass"))  # Class 列
            elif vol_class == 1:
                table.SetValue(row, 1, str("Failed"))  # Class 列
        # 创建并显示节点
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Inf-QC Results")
        table_node.SetAndObserveTable(table)

        # 为 vtkMRMLTableNode 设置列名和数据
        for col_idx in range(table.GetNumberOfColumns()):
            column = table.GetColumn(col_idx)
            table_node.AddColumn(column)  # 将列添加到 vtkMRMLTableNode

        # 刷新 Slicer 界面
        slicer.app.processEvents()

    # ###############
    """ MRI的QC方法"""

    # ###############
    def _process_MC_qc(self, inputVolume, mask, progress_callback=None):

        # 新增 React-QC 处理逻辑
        inputFilePath = inputVolume.GetStorageNode().GetFullNameFromFileName()

        mri_path = inputFilePath

        python_executable = os.path.join(current_dir, "stochastic_MC_QC/venv/Scripts/python")
        # print("Python executable is: ", python_executable)
        # test_script = "QCNet/Inf_QC/predict_from_saved_1.py"
        test_script = os.path.join(current_dir, "stochastic_MC_QC/infer_onnx.py")

        # print("Python test_script is: ", test_script)
        print("mri_path:", mri_path)
        # 启动子进程并传递数据
        # result = subprocess.run(
        #     [python_executable, test_script],  # 子进程命令（此处调用另一个 Python 脚本）
        #     input=mri_path,  # 输入数据（通过 stdin 传递）
        #     text=True,  # 以文本模式处理输入输出
        #     capture_output=True  # 捕获 stdout 和 stderr
        # )
        #
        # # ##########################
        # # 获取子进程的输出
        # if result.returncode == 0:
        #     print("子进程返回结果:", result.stdout)
        #     # print(type(int(result.stdout)))
        #     get_result = result.stdout  # 这是一个结果列表,返回String,如"路径,类别(artifact\pass),置信度"
        #     # print(get_result.split(",")[-2], get_result.split(",")[-1])
        #     QC_type = get_result.split(",")[-2]
        #     QC_probility = get_result.split(",")[-1]
        #
        #     print("类别:", QC_type)
        #     print("置信度:", QC_probility)
        # else:
        #     print("子进程执行出错:", result.stderr)
        #     print("=== STDERR ===")
        #     print(result.stderr)
        #     print("=== STDOUT ===")
        #     print(result.stdout)

        # ######################

        env = {
            "PATH": os.path.join(current_dir, "stochastic_MC_QC/venv/Scripts") + os.pathsep + os.environ.get("PATH",
                                                                                                             ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # Windows必需
            "TEMP": os.environ.get("TEMP", ""),
            "PYTHONHOME": "",  # 强制清空
            "PYTHONPATH": "",  # 强制清空
        }

        result = subprocess.run(
            [python_executable, test_script],
            input=mri_path,
            text=True,
            capture_output=True,
            env=env  # 关键：使用自定义环境变量
        )

        QC_type = None
        QC_probility = None

        if result.returncode == 0:
            print("子进程返回结果:", result.stdout)
            # print(type(int(result.stdout)))
            get_result = result.stdout  # 这是一个结果列表,返回String,如"路径,类别(artifact\pass),置信度"
            # print(get_result.split(",")[-2], get_result.split(",")[-1])
            QC_type = get_result.split(",")[-2]
            QC_probility = get_result.split(",")[-1]

            print("类别:", QC_type)
            print("置信度:", QC_probility)
        else:
            print("子进程执行出错:", result.stderr)
            print("=== STDERR ===")
            print(result.stderr)
            print("=== STDOUT ===")
            print(result.stdout)

        # self.appendMessage("处理完成！", "success")
        # # self.appendMessage("结果如下:", "success")
        # self.appendMessage(f"判断结果: {QC_type}, 置信度：: {QC_probility}","success")
        return QC_type, QC_probility

    # ################
    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                mask,
                bval,
                bvec,
                progress_callback=None,
                method=None
                ) -> None:

        # import time
        # import logging
        #
        # startTime = time.time()
        # logging.info("Processing started")
        #
        # # 分阶段的进度条
        # if progress_callback:
        #     progress_callback(10, "Loading data...")
        #
        # inputFilePath = inputVolume.GetStorageNode().GetFullNameFromFileName()
        #
        # orig_name = inputFilePath
        #
        # print("OKK")
        #
        # volume_classify_results_list = test_single(
        #     orig_name,
        #     mask, bval, bvec, model,
        #     test_batch_size=2,
        #     # progress_callback = progress_callback
        #     progress_callback=progress_callback
        # )
        # print("volume_classify_results_list")
        # print(volume_classify_results_list)
        # # 只保留label=1的索引列表,表示要drop掉这些
        # self.discard_index_list = [index for index, label in volume_classify_results_list if label == 1]
        #
        # # 调用函数显示表格
        # self.create_slicer_table(volume_classify_results_list)
        #
        # stopTime = time.time()
        # logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")
        #
        # if progress_callback:
        #     progress_callback(100, "Done!")

        if method == '3D-QCNet':
            self._process_3d_qcnet(inputVolume, mask, bval, bvec, progress_callback=progress_callback)
        elif method == 'React-QC':
            self._process_react_qc(inputVolume, mask, bval, bvec, progress_callback=progress_callback)
        elif method == 'Inf-QC':
            self._process_inf_qc(inputVolume, mask, bval, bvec, progress_callback=progress_callback)

    def process_MRI(self,
                    inputVolume: vtkMRMLScalarVolumeNode,
                    mask,
                    progress_callback=None,
                    method=None
                    #  ) -> None: 因为要返回一个值,所以这里把-> None去掉了,加上None表示没有返回值
                    ):

        if method == 'Light-Artifact-QC':
            # self._process_3d_qcnet(inputVolume, mask,progress_callback=progress_callback)
            print("Processing Light-Artifact")
        elif method == 'Stochastic-MC-QC':
            QC_type, QC_Prob = self._process_MC_qc(inputVolume, mask)
            return QC_type, QC_Prob

        elif method == 'MRI-QC-3':
            # self._process_inf_qc(inputVolume, mask,progress_callback=progress_callback)
            print("Processing MRI-QC-3")

    # 这里是定义执行saveDiscard按钮的逻辑,例如我将数据保存到某个目录下
    def process_saveDiscard(self, origin_dwi_path,
                            bval_path,
                            bvec_path,
                            save_path,
                            ) -> None:

        import time
        import logging

        startTime = time.time()
        logging.info("Processing started")

        dwi, affine = load_nifti(origin_dwi_path, return_img=False)
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)

        print("bvals:", bvals.shape)
        print("bvecs:", bvecs.shape)

        # 这里传入discard_index,然后剔除dwi中对应的index数据
        print("保留的index:", self.discard_index_list)
        discard_dwi = np.delete(dwi, self.discard_index_list, axis=3)
        print("dwi.shape", discard_dwi.shape)
        discard_bval = np.delete(bvals, self.discard_index_list, axis=0)
        print("bval.shape", discard_bval.shape)
        discard_bvec = np.delete(bvecs, self.discard_index_list, axis=0)
        print("bvec.shape", discard_bvec.shape)
        #
        print("执行成功啦!!!!")

        # 保存文件逻辑
        save_nifti(os.path.join(save_path, "discard_dwi.nii.gz"), discard_dwi, affine)

        np.savetxt(os.path.join(save_path, "discard_dwi.bval"), discard_bval.reshape(1, -1), fmt="%.6f")

        with open(os.path.join(save_path, "discard_dwi.bvec"), "w") as f:
            for row in discard_bvec.T:
                f.write(" ".join(f"{x:.10f}" for x in row) + "\n")

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")
