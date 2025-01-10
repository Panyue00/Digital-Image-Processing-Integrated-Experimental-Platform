import sys
import cv2
import os
from ultralytics import YOLO
import numpy as np
import torch
import torchvision.models.segmentation as models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
from torchvision.transforms import functional as F
import easyocr
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import dip

# ==================== 核心功能函数 ====================
# 初始化YOLO模型
model = YOLO("yolov5s.pt")  # 加载YOLOv5模型

# 在UI中添加一个ComboBox控件，用户选择滤波器
def setup_filter_combobox():
    """设置滤波器选择的下拉菜单"""
    ui.comboBox_filter.addItem("均值滤波")
    ui.comboBox_filter.addItem("高斯滤波")
    ui.comboBox_filter.addItem("Sobel边缘检测")
    ui.comboBox_filter.addItem("拉普拉斯滤波")
    ui.comboBox_filter.addItem("中值滤波")
    ui.comboBox_filter.addItem("傅里叶变换")
    ui.comboBox_filter.addItem("理想低通滤波器")
    ui.comboBox_filter.addItem("高斯高通滤波器")
    ui.comboBox_filter.addItem("腐蚀")
    ui.comboBox_filter.addItem("膨胀")
    ui.comboBox_filter.addItem("开运算")
    ui.comboBox_filter.addItem("闭运算")
    ui.comboBox_filter.addItem("梯度运算")
    ui.comboBox_filter.addItem("全局阈值分割")
    ui.comboBox_filter.addItem("自适应阈值分割")
    ui.comboBox_filter.addItem("ostu阈值法")
    ui.comboBox_filter.addItem("canny边缘检测")
    ui.comboBox_filter.addItem("deeplabv3+")
    ui.comboBox_filter.addItem("maskr_cnn")
    
# 加载 Mask R-CNN 模型
mask_rcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn_model.eval()

def apply_mask_rcnn(image, model, threshold=0.1):
    """
    使用 Mask R-CNN 对输入图像进行实例分割
    :param image: 输入图像 (OpenCV 格式, RGB)
    :param model: Mask R-CNN 模型
    :param threshold: 检测阈值
    :return: 标注结果图像
    """
    # 将输入图像转换为 Tensor 并添加 batch 维度
    img_tensor = F.to_tensor(image).unsqueeze(0)

    # 推理
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # 解析预测结果
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    masks = predictions['masks']

    # 在原图像上绘制结果
    annotated_image = image.copy()
    for i in range(len(scores)):
        if scores[i] > threshold:
            # 获取边界框
            box = boxes[i].cpu().numpy().astype(int)
            cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # 获取分割掩码
            mask = masks[i, 0].cpu().numpy()
            mask = (mask > threshold).astype(np.uint8) * 255

            # 将掩码覆盖到图像上
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[:, :, 1] = mask  # 绿色通道
            annotated_image = cv2.addWeighted(annotated_image, 1, colored_mask, 0.5, 0)

    return annotated_image

# 加载 DeepLabV3+ 模型
def load_deeplab_model():
    model = models.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model


deeplab_model = load_deeplab_model()  # 程序启动时加载模型


def apply_segmentation(image, model):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = torch.argmax(output, dim=0).byte().cpu().numpy()
    return decode_segmentation(output_predictions)


def decode_segmentation(segmentation):
    colormap = np.array([
        [0, 0, 0],         # 背景
        [128, 0, 0],       # 飞机
        [0, 128, 0],       # 自行车
        [128, 128, 0],     # 鸟
        [0, 0, 128],       # 船
        [128, 0, 128],     # 瓶子
        [0, 128, 128],     # 公车
        [128, 128, 128],   # 汽车
        [64, 0, 0],        # 猫
        [192, 0, 0],       # 椅子
        [64, 128, 0],      # 牛
        [192, 128, 0],     # 餐桌
        [64, 0, 128],      # 狗
        [192, 0, 128],     # 马
        [64, 128, 128],    # 摩托车
        [192, 128, 128],   # 人
        [0, 64, 0],        # 花盆
        [128, 64, 0],      # 羊
        [0, 192, 0],       # 沙发
        [128, 192, 0],     # 火车
        [0, 64, 128]       # 显示器
    ])
    # 映射分割结果到颜色
    colored_segmentation = colormap[segmentation]
    return colored_segmentation.astype(np.uint8)


def update_slider_label():
    """更新滑块值显示"""
    value = ui.horizontalSlider.value()
    ui.label_slider_value.setText(f"核大小: {value}x{value}")
    
def update_slider_label_2():
    """更新滑块值显示"""
    value = ui.horizontalSlider.value()
    ui.label_slider_value_2.setText(f"截止频率: {value}")

def update_slider_label_3():
    """更新滑块值显示"""
    value = ui.horizontalSlider.value()
    ui.label_slider_value_3.setText(f"元素大小: {value}")
  
def update_slider_label_4():
    """更新滑块值显示"""
    value = ui.horizontalSlider.value()
    ui.label_slider_value_4.setText(f"阈值: {value}")            
def apply_mean_filter(image, kernel_size):
    """
    应用均值滤波
    :param image: 输入图像 (OpenCV 格式)
    :param kernel_size: 滤波核大小
    :return: 滤波后的图像
    """
    kernel = (kernel_size, kernel_size)  # 核为正方形
    return cv2.blur(image, kernel)

def apply_gaussian_filter(image, kernel_size):
    """
    应用高斯滤波
    :param image: 输入图像 (OpenCV 格式)
    :param kernel_size: 滤波核大小，需为奇数
    :return: 滤波后的图像
    """
    if kernel_size % 2 == 0:  # 确保核大小为奇数
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_sobel_edge_detection(image):
    """应用 Sobel 边缘检测"""
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 计算 Sobel 边缘检测，水平和垂直方向
    grad_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1)

    # 计算梯度的绝对值，并转换回 uint8 格式
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)

    # 合并两个方向的梯度
    sobel_edges = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    return sobel_edges

def apply_laplacian_filter(image):
    """应用拉普拉斯滤波"""
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 应用拉普拉斯滤波
    laplacian = cv2.Laplacian(gray_image, cv2.CV_16S, ksize=3)

    # 将结果转换为 uint8 格式
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    return laplacian_abs

def apply_median_filter(image, kernel_size):
    """应用中值滤波"""
    return cv2.medianBlur(image, kernel_size)

def apply_fourier_transform(image):
    """
    应用傅里叶变换，并返回频谱图
    :param image: 输入图像（OpenCV 格式，单通道灰度图像）
    :return: 频谱图
    """
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 执行傅里叶变换
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # 移动零频率分量到中心
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum

def apply_ideal_lowpass_filter(image, cutoff_frequency):
    """
    应用理想低通滤波器
    :param image: 输入图像（OpenCV 格式，单通道灰度图像）
    :param cutoff_frequency: 截止频率（整数，决定滤波器的半径）
    :return: 滤波后的图像
    """
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # 获取图像尺寸
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2  # 中心点

    # 傅里叶变换
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建理想低通滤波器
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if distance <= cutoff_frequency:
                mask[i, j] = 1

    # 应用滤波器
    filtered_dft = dft_shift * mask

    # 傅里叶逆变换
    dft_ishift = np.fft.ifftshift(filtered_dft)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back

def apply_gaussian_highpass_filter(image, cutoff_frequency):
    """
    应用高斯高通滤波器 (灰度图像)
    :param image: 输入灰度图像 (OpenCV 格式)
    :param cutoff_frequency: 截止频率
    :return: 高通滤波后的灰度图像
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 创建与图像大小一致的坐标网格
    x = np.linspace(-ccol, ccol - 1, cols)
    y = np.linspace(-crow, crow - 1, rows)
    X, Y = np.meshgrid(x, y)

    # 计算高斯高通掩膜
    D = np.sqrt(X**2 + Y**2)
    gaussian_highpass = 1 - np.exp(-(D**2) / (2 * (cutoff_frequency**2)))

    # DFT 变换
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # 应用高斯高通滤波器
    dft_filtered = dft_shift * gaussian_highpass

    # 逆傅里叶变换
    dft_ishift = np.fft.ifftshift(dft_filtered)
    filtered_image = np.abs(np.fft.ifft2(dft_ishift))

    return filtered_image

def apply_erosion(image, kernel_size):
    """
    应用腐蚀操作 (直接在彩色图像上)
    :param image: 输入图像 (OpenCV 格式)
    :param kernel_size: 结构元素的大小，用于腐蚀操作
    :return: 腐蚀后的图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 创建结构元素
    return cv2.erode(image, kernel, iterations=1)  # 执行腐蚀操作

def apply_dilation(image, kernel_size):
    """
    应用膨胀操作 (直接在彩色图像上)
    :param image: 输入图像 (OpenCV 格式)
    :param kernel_size: 结构元素的大小，用于膨胀操作
    :return: 膨胀后的图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 创建结构元素
    return cv2.dilate(image, kernel, iterations=1)  # 执行膨胀操作

def apply_opening(image, kernel_size):
    """
    应用开运算 (腐蚀后再膨胀)
    :param image: 输入图像 (OpenCV 格式)
    :param kernel_size: 结构元素的大小，用于腐蚀和膨胀操作
    :return: 开运算后的图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 创建结构元素
    eroded_image = cv2.erode(image, kernel, iterations=1)  # 先进行腐蚀
    opened_image = cv2.dilate(eroded_image, kernel, iterations=1)  # 再进行膨胀
    return opened_image

def apply_closing(image, kernel_size):
    """
    应用闭运算 (膨胀后再腐蚀)
    :param image: 输入图像 (OpenCV 格式)
    :param kernel_size: 结构元素的大小，用于膨胀和腐蚀操作
    :return: 闭运算后的图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 创建结构元素
    dilated_image = cv2.dilate(image, kernel, iterations=1)  # 先进行膨胀
    closed_image = cv2.erode(dilated_image, kernel, iterations=1)  # 再进行腐蚀
    return closed_image

def apply_morphological_gradient(image, kernel_size):
    """
    应用形态学梯度运算 (膨胀 - 腐蚀)
    :param image: 输入图像 (OpenCV 格式)
    :param kernel_size: 结构元素的大小
    :return: 形态学梯度后的图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 创建结构元素
    dilated_image = cv2.dilate(image, kernel, iterations=1)  # 先进行膨胀
    eroded_image = cv2.erode(image, kernel, iterations=1)  # 再进行腐蚀
    gradient_image = cv2.subtract(dilated_image, eroded_image)  # 膨胀图像与腐蚀图像的差异
    return gradient_image

def apply_global_threshold_rgb(image, threshold_value):
    """
    应用全局阈值分割 (RGB 图像)
    :param image: 输入彩色图像 (OpenCV 格式)
    :param threshold_value: 阈值
    :return: 处理后的二值化图像
    """
    # 将图像拆分为三个通道
    r, g, b = cv2.split(image)
    
    # 对每个通道应用阈值分割
    _, r_bin = cv2.threshold(r, threshold_value, 255, cv2.THRESH_BINARY)
    _, g_bin = cv2.threshold(g, threshold_value, 255, cv2.THRESH_BINARY)
    _, b_bin = cv2.threshold(b, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 合并三个二值通道
    binary_image = cv2.merge([r_bin, g_bin, b_bin])
    return binary_image

def apply_adaptive_threshold_rgb(image, block_size, C):
    """
    应用自适应阈值分割 (RGB 图像)
    :param image: 输入彩色图像 (OpenCV 格式)
    :param block_size: 邻域大小 (必须为奇数)
    :param C: 阈值常量
    :return: 处理后的图像
    """
    # 将图像拆分为三个通道
    r, g, b = cv2.split(image)

    # 对每个通道应用自适应阈值
    r_adaptive = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    g_adaptive = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    b_adaptive = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

    # 合并三个通道
    adaptive_binary_image = cv2.merge([r_adaptive, g_adaptive, b_adaptive])
    return adaptive_binary_image

def apply_otsu_threshold_rgb(image):
    """
    应用 Otsu 阈值分割 (RGB 图像)
    :param image: 输入彩色图像 (OpenCV 格式)
    :return: 处理后的图像
    """
    # 将图像拆分为三个通道
    r, g, b = cv2.split(image)

    # 对每个通道应用 Otsu 阈值分割
    _, r_otsu = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, g_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b_otsu = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 合并三个通道
    otsu_binary_image = cv2.merge([r_otsu, g_otsu, b_otsu])
    return otsu_binary_image

def apply_canny_edge_detection(image, low_threshold=100, high_threshold=200):
    """
    应用 Canny 边缘检测
    :param image: 输入图像 (OpenCV 格式)
    :param low_threshold: Canny 边缘检测的低阈值
    :param high_threshold: Canny 边缘检测的高阈值
    :return: 边缘检测后的图像
    """
    # 转换为灰度图像，因为 Canny 边缘检测通常需要灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用 Canny 边缘检测
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    # 将边缘检测结果转换为三通道图像（黑白图像的 RGB 表现）
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return edges_rgb

# 应用滤波器并显示结果
def apply_filter(img, filter_type, kernel_size):

    if filter_type == '均值滤波':
        return apply_mean_filter(img, kernel_size)
    elif filter_type == '高斯滤波':
        return apply_gaussian_filter(img, kernel_size)
    elif filter_type == 'Sobel边缘检测':
        return apply_sobel_edge_detection(img)
    elif filter_type == '拉普拉斯滤波':
        return apply_laplacian_filter(img)
    elif filter_type == '中值滤波':
        return apply_median_filter(img, kernel_size)
    elif filter_type == '傅里叶变换':
        return apply_fourier_transform(img)
    elif filter_type == '理想低通滤波器':
        return apply_ideal_lowpass_filter(img, kernel_size)
    elif filter_type == '高斯高通滤波器':
        return apply_gaussian_highpass_filter(img, kernel_size)
    elif filter_type == '腐蚀':
        return apply_erosion(img, kernel_size)
    elif filter_type == '膨胀':
        return apply_dilation(img, kernel_size)
    elif filter_type == '开运算':
        return apply_opening(img, kernel_size)
    elif filter_type == '闭运算':
        return apply_closing(img, kernel_size)
    elif filter_type == '梯度运算':
        return apply_morphological_gradient(img, kernel_size)
    elif filter_type == '全局阈值分割':
        return apply_global_threshold_rgb(img, "global", kernel_size)
    elif filter_type == '自适应阈值分割':
        return apply_adaptive_threshold_rgb(img, "adaptive")
    elif filter_type == 'ostu阈值法':
        return apply_otsu_threshold_rgb(img, "otsu")
    elif filter_type == 'canny边缘检测':
        return apply_canny_edge_detection(img)
    elif filter_type == 'deeplabv3+':
        return apply_segmentation(img, deeplab_model)
    elif filter_type == 'maskr_cnn':
        return apply_mask_rcnn(img, mask_rcnn_model)
    else:
        return img  # 如果没有有效的滤波器类型，返回原图像
    
def apply_pipeline(img, filters, kernel_size, threshold1, threshold2):
    """
    按顺序应用多个图像处理步骤（流水线）
    :param image: 输入图像
    :param filters: 包含每个步骤的滤波器（例如: ['mean', 'gaussian']）
    :param kernel_size: 滤波器的大小
    :param threshold1: Canny边缘检测的第一个阈值
    :param threshold2: Canny边缘检测的第二个阈值
    :return: 处理后的图像
    """
    
    filtered_image = img.copy()  # 保留原图像作为初始图像
    
    for filter_type in filters:
        if filter_type == '均值滤波':
            filtered_image = apply_mean_filter(filtered_image, kernel_size)
        elif filter_type == '高斯滤波':
            filtered_image = apply_gaussian_filter(filtered_image, kernel_size)
        elif filter_type == 'Sobel边缘检测':
            filtered_image = apply_sobel_edge_detection(filtered_image)
        elif filter_type == '拉普拉斯滤波':
            filtered_image = apply_laplacian_filter(filtered_image)
        elif filter_type == '中值滤波':
            filtered_image = apply_median_filter(filtered_image, kernel_size)
        elif filter_type == '傅里叶变换':
            filtered_image = apply_fourier_transform(filtered_image)
        elif filter_type == '理想低通滤波器':
            filtered_image = apply_ideal_lowpass_filter(filtered_image, kernel_size)
        elif filter_type == '高斯高通滤波器':
            filtered_image = apply_gaussian_highpass_filter(filtered_image, kernel_size)
        elif filter_type == '腐蚀':
            filtered_image = apply_erosion(filtered_image, kernel_size)
        elif filter_type == '膨胀':
            filtered_image = apply_dilation(filtered_image, kernel_size)
        elif filter_type == '开运算':
            filtered_image = apply_opening(filtered_image, kernel_size)
        elif filter_type == '闭运算':
            filtered_image = apply_closing(filtered_image, kernel_size)
        elif filter_type == '梯度运算':
            filtered_image = apply_morphological_gradient(filtered_image, kernel_size)
        else:
            print(f"未知的滤波器类型: {filter_type}")
    return filtered_image   # 如果没有有效的滤波器类型，返回原图像

def detect_faces(image):
    """
    使用 face_recognition 检测人脸
    :param image: 输入的 RGB 图像 (OpenCV 格式)
    :return: 标注了人脸边框的图像, 检测到的人脸信息 [(top, right, bottom, left)]
    """
    # 转换为 RGB 格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 检测人脸位置
    face_locations = face_recognition.face_locations(rgb_image)
    return face_locations
    
# ==================== 按钮事件函数 ====================

def click_pushButton():
    """选择图片并显示在 QLabel"""
    file_path, _ = QFileDialog.getOpenFileName(
        None, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
    )
    if file_path:
        # 加载图片并显示
        pixmap = QPixmap(file_path)
        ui.label.setPixmap(pixmap)
        ui.label.setScaledContents(True)  # 使图片适应 QLabel 大小

def click_pushButton_3():
    """
    将 label_2 中的图片保存到 output 文件夹
    """
    if ui.label_2.pixmap() is not None:
        # 获取 label_2 显示的图片
        pixmap = ui.label_2.pixmap()

        # 将 QPixmap 转换为 QImage
        image = pixmap.toImage()

        # 创建输出文件夹
        output_folder = "output"
        import os
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 定义保存路径（可以根据需要生成唯一文件名）
        import datetime
        filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = os.path.join(output_folder, filename)

        # 将 QImage 保存到文件
        if image.save(output_path):
            print(f"图片已成功保存到: {output_path}")
        else:
            print("保存失败！")
    else:
        print("label_2 中没有图片，无法保存！")

def click_pushButton_4():
    ui.textBrowser.setPlainText(
        "均值滤波：均值滤波是一种图像平滑技术，它通过计算每个像素邻域区域的平均值来减少图像噪声。"
        "在此操作中，核大小参数用于确定滤波器的尺寸，较大的核可以更有效地平滑图像，但会模糊图像的细节。"
    )
    """点击按钮应用均值滤波并显示结果"""
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 设置数据大小
        img_array = np.array(img_data).reshape((height, width, 4))  # RGBA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值作为滤波核大小并应用均值滤波
        kernel_size = ui.horizontalSlider.value()
        filtered_image = apply_mean_filter(img_rgb, kernel_size)

        # 将处理后的图像转换回 RGB 格式
        filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # 将处理后的图像转换为 QPixmap 并显示在 QLabel
        height, width, channel = filtered_image_rgb.shape
        step = channel * width
        q_img = QImage(filtered_image_rgb.data, width, height, step, QImage.Format_RGB888)
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_5():
    ui.textBrowser.setPlainText(
        "高斯滤波：高斯滤波是一种加权平均滤波方法，通过高斯函数对邻域像素进行加权，"
        "核大小和标准差决定了滤波器的效果。较大的核和标准差会产生更强的平滑效果，适合处理噪声。"
    )
    """点击按钮应用高斯滤波并显示结果"""
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 设置数据大小
        img_array = np.array(img_data).reshape((height, width, 4))  # RGBA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值作为滤波核大小并应用高斯滤波
        kernel_size = ui.horizontalSlider.value()
        filtered_image = apply_gaussian_filter(img_rgb, kernel_size)

        # 将处理后的图像转换回 RGB 格式
        filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # 将处理后的图像转换为 QPixmap 并显示在 QLabel
        height, width, channel = filtered_image_rgb.shape
        step = channel * width
        q_img = QImage(filtered_image_rgb.data, width, height, step, QImage.Format_RGB888)
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)
        
def click_pushButton_6():
    ui.textBrowser.setPlainText(
        "Sobel 边缘检测：Sobel 算子通过计算图像在水平方向和垂直方向的梯度，突出图像中的边缘。"
        "该方法不直接使用核大小参数，但在应用之前通常会先进行平滑处理（如高斯滤波）。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 应用 Sobel 边缘检测
        sobel_edges = apply_sobel_edge_detection(img_rgb)

        # 将处理后的图像转换为 QPixmap 显示
        height, width = sobel_edges.shape
        step = width
        q_img = QImage(sobel_edges.data, width, height, step, QImage.Format_Grayscale8)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)      
        
def click_pushButton_7():
    ui.textBrowser.setPlainText(
        "拉普拉斯滤波：拉普拉斯滤波用于边缘检测，它通过计算图像的二阶导数来突出边缘。"
        "拉普拉斯滤波对噪声敏感，通常需要与平滑操作结合使用。核大小决定了计算的邻域范围，较大的核会平滑更多噪声。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 应用拉普拉斯滤波
        laplacian_image = apply_laplacian_filter(img_rgb)

        # 将处理后的图像转换为 QPixmap 显示
        height, width = laplacian_image.shape
        step = width
        q_img = QImage(laplacian_image.data, width, height, step, QImage.Format_Grayscale8)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)          

def click_pushButton_15():
    ui.textBrowser.setPlainText(
        "中值滤波是一种非线性滤波技术，主要用于图像降噪。它通过将图像中每个像素点的值替换为其邻域像素值的中值，以去除噪声。中值滤波对椒盐噪声特别有效，能够较好地保留图像的边缘信息，避免模糊效果。"
        "较小的核大小适用于细节较多的图像，较大的核大小适用于噪声较多的图像。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)
        img_array = np.array(img_data).reshape((height, width, 4))
        img_rgb = img_array[:, :, :3]

        # 获取滑块值并调整为奇数
        kernel_size = ui.horizontalSlider.value()
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 应用中值滤波
        median_filtered_image = apply_median_filter(img_rgb, kernel_size)

        # 将处理后的图像转换为 QPixmap 显示
        filtered_image_rgb = cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB)
        height, width, channel = filtered_image_rgb.shape
        step = channel * width
        q_img = QImage(filtered_image_rgb.data, width, height, step, QImage.Format_RGB888)

        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_8():
    ui.textBrowser.setPlainText(
        "傅里叶变换：傅里叶变换将图像从时域转换到频域，分析其频率成分。"
        "通过改变截止频率，可以去除频域中的低频或高频成分，从而实现平滑或锐化效果。"
    )
    """点击 button_8 进行傅里叶变换并显示结果"""
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 保留 RGB 通道

        # 应用傅里叶变换
        fourier_result = apply_fourier_transform(img_rgb)

        # 归一化到 0-255 范围，并转换为 QPixmap 显示
        normalized_result = cv2.normalize(fourier_result, None, 0, 255, cv2.NORM_MINMAX)
        result_uint8 = np.uint8(normalized_result)
        height, width = result_uint8.shape
        q_img = QImage(result_uint8.data, width, height, width, QImage.Format_Grayscale8)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_9():
    ui.textBrowser.setPlainText(
        "理想低通滤波：理想低通滤波器通过去除频域中的高频部分，保留低频成分，通常用于去除图像中的噪声。"
        "截止频率决定了滤波器去除的频率范围，较低的截止频率会去除更多的高频信息，导致图像更加平滑。"
    )
    """点击 button9 进行理想低通滤波并显示结果"""
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 保留 RGB 通道

        # 应用理想低通滤波器
        cutoff_frequency = ui.horizontalSlider.value()  # 获取滑块值作为截止频率
        filtered_image = apply_ideal_lowpass_filter(img_rgb, cutoff_frequency)

        # 归一化到 0-255 范围，并转换为 QPixmap 显示
        normalized_result = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
        result_uint8 = np.uint8(normalized_result)
        height, width = result_uint8.shape
        q_img = QImage(result_uint8.data, width, height, width, QImage.Format_Grayscale8)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)
        
def click_pushButton_10():
    ui.textBrowser.setPlainText(
        "高斯高通滤波：高斯高通滤波器通过去除低频成分来保留图像的高频细节，通常用于图像锐化或边缘检测。"
        "截止频率确定了滤波器的效果，较高的截止频率会保留更多的图像细节，但也可能增强噪声。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为灰度图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

        # 获取滑块值，作为截止频率
        cutoff_frequency = ui.horizontalSlider.value()

        # 应用高斯高通滤波
        filtered_image = apply_gaussian_highpass_filter(img_gray, cutoff_frequency)

        # 将处理后的图像转换回 RGB 格式
        filtered_image_rgb = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # 将处理后的图像转换为 QPixmap 显示
        height, width, channel = filtered_image_rgb.shape
        step = channel * width
        q_img = QImage(filtered_image_rgb.data, width, height, step, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_11():
    ui.textBrowser.setPlainText(
        "腐蚀操作：腐蚀操作使用结构元素来减小物体的尺寸，通常用于去除小的噪点或细小物体。"
        "结构元素的形状（如圆形、方形等）和大小会影响腐蚀操作的效果，较小的元素适合处理细小噪声。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值，作为结构元素的大小
        kernel_size = ui.horizontalSlider.value()

        # 应用腐蚀操作
        eroded_image = apply_erosion(img_rgb, kernel_size)

        # 将处理后的图像转换回 RGB 格式
        eroded_image_rgb = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB)

        # 将处理后的图像转换为 QPixmap 显示
        height, width, channel = eroded_image_rgb.shape
        step = channel * width
        q_img = QImage(eroded_image_rgb.data, width, height, step, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_12():
    ui.textBrowser.setPlainText(
        "膨胀操作：膨胀操作使用结构元素来扩大物体的尺寸，常用于填补小孔或连接断裂部分。"
        "较大的结构元素能够连接更大的物体或填补更大的孔洞。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值，作为结构元素的大小
        kernel_size = ui.horizontalSlider.value()

        # 应用膨胀操作
        dilated_image = apply_dilation(img_rgb, kernel_size)

        # 将处理后的图像转换回 RGB 格式
        dilated_image_rgb = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB)

        # 将处理后的图像转换为 QPixmap 显示
        height, width, channel = dilated_image_rgb.shape
        step = channel * width
        q_img = QImage(dilated_image_rgb.data, width, height, step, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_13():
    ui.textBrowser.setPlainText(
        "开运算：开运算是腐蚀后再膨胀的一种形态学操作，通常用于去除小的噪点或断裂部分。"
        "结构元素的大小决定了操作效果，较小的元素可以去除细小噪点，较大的元素适合连接物体。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值，作为结构元素的大小
        kernel_size = ui.horizontalSlider.value()

        # 应用开运算操作
        opened_image = apply_opening(img_rgb, kernel_size)

        # 将处理后的图像转换回 RGB 格式
        opened_image_rgb = cv2.cvtColor(opened_image, cv2.COLOR_BGR2RGB)

        # 将处理后的图像转换为 QPixmap 显示
        height, width, channel = opened_image_rgb.shape
        step = channel * width
        q_img = QImage(opened_image_rgb.data, width, height, step, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_14():
    ui.textBrowser.setPlainText(
        "闭运算：闭运算是膨胀后再腐蚀的一种形态学操作，适用于填补物体中的小孔或连接物体的断裂部分。"
        "结构元素的形状和大小决定了填补的效果，较大的元素适合修复较大孔洞。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值，作为结构元素的大小
        kernel_size = ui.horizontalSlider.value()

        # 应用闭运算操作
        closed_image = apply_closing(img_rgb, kernel_size)

        # 将处理后的图像转换回 RGB 格式
        closed_image_rgb = cv2.cvtColor(closed_image, cv2.COLOR_BGR2RGB)

        # 将处理后的图像转换为 QPixmap 显示
        height, width, channel = closed_image_rgb.shape
        step = channel * width
        q_img = QImage(closed_image_rgb.data, width, height, step, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)
        
def click_pushButton_16():
    ui.textBrowser.setPlainText(
        "形态学梯度：形态学梯度通过计算膨胀和腐蚀的差值来突出图像中的边缘。"
        "结构元素的选择对于梯度提取的效果有重要影响，较大的结构元素会增强物体的轮廓。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值，作为结构元素的大小
        kernel_size = ui.horizontalSlider.value()

        # 应用形态学梯度操作
        gradient_image = apply_morphological_gradient(img_rgb, kernel_size)

        # 将处理后的图像转换回 RGB 格式
        gradient_image_rgb = cv2.cvtColor(gradient_image, cv2.COLOR_BGR2RGB)

        # 将处理后的图像转换为 QPixmap 显示
        height, width, channel = gradient_image_rgb.shape
        step = channel * width
        q_img = QImage(gradient_image_rgb.data, width, height, step, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_17():
    ui.textBrowser.setPlainText(
        "全局阈值分割：全局阈值分割通过设定一个固定的阈值来区分前景和背景。"
        "选择合适的阈值非常重要，过高或过低的阈值会导致分割失败。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值，作为全局阈值
        threshold_value = ui.horizontalSlider.value()

        # 应用全局阈值分割
        binary_image = apply_global_threshold_rgb(img_rgb, threshold_value)
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB)
        # 将二值化图像转换为 QPixmap 显示
        height, width, _ = binary_image.shape
        q_img = QImage(binary_image.data, width, height, width * 3, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_18():
    ui.textBrowser.setPlainText(
        "自适应阈值分割：自适应阈值分割根据图像的局部特征动态调整阈值，适用于光照不均的图像。"
        "通过自适应调整阈值，能够更好地处理不同区域的图像内容。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 获取滑块值，作为自适应阈值的参数
        block_size = ui.horizontalSlider.value() * 2 + 1  # 滑块值调整为邻域大小（奇数）
        C = 5  # 可以根据需要调整

        # 应用自适应阈值分割
        adaptive_binary_image = apply_adaptive_threshold_rgb(img_rgb, block_size, C)
        adaptive_binary_image = cv2.cvtColor( adaptive_binary_image, cv2.COLOR_BGR2RGB)
        # 将二值化图像转换为 QPixmap 显示
        height, width, _ = adaptive_binary_image.shape
        q_img = QImage(adaptive_binary_image.data, width, height, width * 3, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_19():
    ui.textBrowser.setPlainText(
        "Otsu 阈值分割：Otsu 阈值分割通过最大化类间方差自动选择最佳阈值，通常用于灰度图像的分割。"
        "该方法不需要人工设定阈值，自动根据图像的灰度分布来确定最佳分割点。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 应用 Otsu 阈值分割
        otsu_binary_image = apply_otsu_threshold_rgb(img_rgb)
        otsu_binary_image = cv2.cvtColor(otsu_binary_image, cv2.COLOR_BGR2RGB)
        # 将二值化图像转换为 QPixmap 显示
        height, width, _ = otsu_binary_image.shape
        q_img = QImage(otsu_binary_image.data, width, height, width * 3, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_20():
    ui.textBrowser.setPlainText(
        "Canny 边缘检测：Canny 边缘检测是一种多阶段的边缘检测算法，能够精确提取图像中的边缘信息。"
        "算法中使用的两个阈值决定了边缘的强度范围，合理选择阈值可以提高边缘检测的效果。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 为了获取数据，设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 三个通道

        # 应用 Canny 边缘检测
        edges_rgb = apply_canny_edge_detection(img_rgb)

        # 将处理后的图像转换为 QPixmap 显示
        height, width, _ = edges_rgb.shape
        q_img = QImage(edges_rgb.data, width, height, width * 3, QImage.Format_RGB888)

        # 在 label_2 上显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)

def click_pushButton_21():
    ui.textBrowser.setPlainText(
        "DeepLabV3+ 图像分割：DeepLabV3+ 是一种基于深度学习的语义分割模型，采用深度卷积神经网络对图像进行像素级分类。"
        "该模型在图像分割中表现优异，能够识别并分割图像中的不同物体。"
        "DeepLabV3+ 中使用了空洞卷积（dilated convolution）来增加感受野，从而提高对复杂场景的分割精度。"
        "分割结果中的每个像素将被分类为一个特定的物体类别，通常用于目标检测、自动驾驶、医学影像分析等领域。"
    )
    if ui.label.pixmap() is not None:
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        img_data = original_image.bits()
        img_data.setsize(height * width * 4)
        img_array = np.array(img_data).reshape((height, width, 4))
        img_rgb = img_array[:, :, :3]

        segmented_image = apply_segmentation(img_rgb, deeplab_model)
        segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

        height, width, channel = segmented_image_rgb.shape
        step = channel * width
        q_img = QImage(segmented_image_rgb.data, width, height, step, QImage.Format_RGB888)

        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)
    else:
        QMessageBox.warning(ui, "警告", "未加载图像，请加载图像后再进行处理。")

def click_pushButton_22():
    ui.textBrowser.setPlainText(
        "MASK R-CNN 图像分割：MASK R-CNN 是一种基于深度学习的实例分割算法，它在 Faster R-CNN 的基础上增加了一个分支，用于生成物体的二进制掩码（mask）。"
        "与语义分割不同，实例分割不仅区分不同的物体类别，还能够区分同一类别的不同实例。"
        "MASK R-CNN 在检测物体的同时，还能够生成精确的物体边界掩码，适用于自动驾驶、视频监控、医学图像等任务。"
        "该方法通过区域提议网络（RPN）来检测物体，并通过卷积神经网络（CNN）生成每个物体的像素级掩码。"
    )
    if ui.label.pixmap() is not None:
        # 获取原图像
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)
        img_array = np.array(img_data).reshape((height, width, 4))  # BGRA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 通道

        # 使用 Mask R-CNN 模型
        segmented_image = apply_mask_rcnn(img_rgb, mask_rcnn_model)

        # 转换为 QPixmap 显示
        segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        height, width, channel = segmented_image_rgb.shape
        step = channel * width
        q_img = QImage(segmented_image_rgb.data, width, height, step, QImage.Format_RGB888)

        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)


                          
# 批量处理图片
def batch_process_images():
    ui.textBrowser.setPlainText(
        "批量处理图片：批量处理是指一次性对多个图像进行相同或类似的处理操作。"
        "该功能可以自动遍历指定文件夹中的所有图像文件，并对每张图片进行处理（如滤波、边缘检测、分割等）。"
        "批量处理能够大大提高处理效率，适用于需要对大量图像进行相同操作的场景，例如图像数据预处理、特征提取等。"
    )
    """批量选择图片并处理"""
    # 获取用户选择的滤波器类型
    filter_type = ui.comboBox_filter.currentText().lower()  # 获取选择的滤波器类型（均值或高斯）
    print(f"当前选择的滤波器类型: {filter_type}")  # 打印选择的滤波器类型
    if filter_type not in ['均值滤波', '高斯滤波','sobel边缘检测','拉普拉斯滤波','中值滤波','傅里叶变换','理想低通滤波器','高斯高通滤波器','腐蚀','膨胀','开运算','闭运算','梯度运算','全局阈值分割','自适应阈值分割','ostu阈值法','canny边缘检测','deeplabv3+','maskr_cnn']:
        print("请选择有效的滤波器类型")
        return  # 如果没有选择有效的滤波器，返回

    # 打开文件对话框，允许多选
    file_paths, _ = QFileDialog.getOpenFileNames(ui.centralwidget, "选择图片", "", "所有文件 (*);;图片文件 (*.png *.jpg *.jpeg)")

    if file_paths:
        # 获取滑块值作为滤波核大小
        kernel_size = ui.horizontalSlider.value()

        # 创建一个保存目录
        save_dir = QFileDialog.getExistingDirectory(ui.centralwidget, "选择保存目录")
        if not save_dir:
            return  # 如果没有选择保存目录，则退出

            # 初始化进度条
        ui.progressBar.setRange(0, len(file_paths))  # 设置进度条范围为文件数量
        ui.progressBar.setValue(0)  # 初始值为0
    
        for i,file_path in enumerate(file_paths):
            print(f"正在读取文件: {file_path}")  # 打印正在处理的文件路径
            img = cv2.imread(file_path)  # 读取图片
            if img is None:
                print(f"无法读取图片: {file_path}")
                continue  # 如果图片无法读取，跳过该图片

            # 应用选中的滤波器
            filtered_img = apply_filter(img, filter_type, kernel_size)

            # 获取文件名并创建新的保存路径
            base_name = os.path.basename(file_path)
            save_path = os.path.join(save_dir, f"filtered_{base_name}")

            # 保存处理后的图片
            cv2.imwrite(save_path, filtered_img)
            ui.progressBar.setValue(i + 1)  # 设置进度条当前值 
            
        print(f"批量处理完成，处理后的图片已保存到 {save_dir}")


    
# 初始化一个列表来追踪勾选的顺序
checked_filters_order = []

def on_checkbox_state_changed(checkbox_name, checked):
    global checked_filters_order
    
    if checked:
        # 如果勾选，则将对应的滤波器按顺序添加到列表中
        if checkbox_name not in checked_filters_order:
            checked_filters_order.append(checkbox_name)
    else:
        # 如果取消勾选，则将对应的滤波器从列表中移除
        if checkbox_name in checked_filters_order:
            checked_filters_order.remove(checkbox_name)
    
    # 打印当前选中的滤波器
    ui.textBrowser.append(f"当前选中的滤波器: {checked_filters_order}")


def process_single_image():
    # 获取滑块值作为滤波核大小和阈值
    kernel_size = ui.horizontalSlider.value()

    # 获取label上的图像
    pixmap = ui.label.pixmap()
    if pixmap is None:
        ui.textBrowser.append("label上没有图片")
        return

    # 将QPixmap转换为OpenCV格式的图像
    image = pixmap.toImage()
    width, height = image.width(), image.height()
    img_data = image.bits()
    img_data.setsize(height * width * 4)
    img_array = np.array(img_data).reshape((height, width, 4))  # RGBA格式
    img_rgb = img_array[:, :, :3]  # 只保留RGB通道

    # 应用选中的流水线操作
    filtered_img = apply_pipeline(img_rgb, checked_filters_order, kernel_size)

    if filtered_img is None or filtered_img.size == 0:
        ui.textBrowser.append("处理后的图像为空")
        return

    # 将处理后的图像转换为 RGB 格式
    filtered_image_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
    height, width, channel = filtered_image_rgb.shape
    step = channel * width
    q_img = QImage(filtered_image_rgb.data, width, height, step, QImage.Format_RGB888)

    # 显示在label2上
    ui.label_2.setPixmap(QPixmap.fromImage(q_img))
    ui.label_2.setScaledContents(True)

    ui.textBrowser.append("图像处理完成并显示在label2上")


def apply_pipeline(img, filters, kernel_size):
    ui.textBrowser.setPlainText(
        "流水线操作：该功能将一系列图像处理步骤按顺序依次应用在同一图像上。"
        "每个处理步骤会基于前一个步骤的结果进行修改，直到所有选中的操作完成。"
        "这使得图像可以经过一系列的滤波、变换、检测等处理，形成最终的图像效果。"
    )
    processed_img = img.copy()

    # 按照用户勾选的顺序依次应用每个滤波器
    for i, filter_name in enumerate(filters):
        ui.textBrowser.append(f"处理第 {i+1} 步: {filter_name}")

        # 保证所有操作都在 uint8 类型下执行
        processed_img = np.array(processed_img, dtype=np.uint8)

        if filter_name == '均值滤波':
            ui.textBrowser.append(f"应用均值滤波，核大小: {kernel_size}")
            processed_img = cv2.blur(processed_img, (kernel_size, kernel_size))
        elif filter_name == '高斯滤波':
            ui.textBrowser.append(f"应用高斯滤波，核大小: {kernel_size}")
            processed_img = cv2.GaussianBlur(processed_img, (kernel_size, kernel_size), 0)
        elif filter_name == 'Sobel边缘检测':
            ui.textBrowser.append(f"应用Sobel边缘检测，核大小: {kernel_size}")
            grad_x = cv2.Sobel(processed_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(processed_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
            processed_img = cv2.magnitude(grad_x, grad_y)
            processed_img = np.array(processed_img, dtype=np.uint8)
        elif filter_name == '拉普拉斯滤波':
            ui.textBrowser.append(f"应用拉普拉斯滤波，核大小: {kernel_size}")
            processed_img = cv2.Laplacian(processed_img, cv2.CV_64F, ksize=kernel_size)
            processed_img = np.array(processed_img, dtype=np.uint8)
        elif filter_name == '中值滤波':
            ui.textBrowser.append(f"应用中值滤波，核大小: {kernel_size}")
            processed_img = cv2.medianBlur(processed_img, kernel_size)
        elif filter_name == '腐蚀':
            ui.textBrowser.append(f"应用腐蚀，核大小: {kernel_size}")
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            processed_img = cv2.erode(processed_img, kernel, iterations=1)
        elif filter_name == '膨胀':
            ui.textBrowser.append(f"应用膨胀，核大小: {kernel_size}")
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            processed_img = cv2.dilate(processed_img, kernel, iterations=1)
        elif filter_name == '开运算':
            ui.textBrowser.append(f"应用开运算，核大小: {kernel_size}")
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)
        elif filter_name == '闭运算':
            ui.textBrowser.append(f"应用闭运算，核大小: {kernel_size}")
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)
        elif filter_name == '梯度运算':
            ui.textBrowser.append(f"应用梯度运算，核大小: {kernel_size}")
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_GRADIENT, kernel)
        elif filter_name == '理想低通滤波器':
            ui.textBrowser.append(f"应用理想低通滤波器，截止频率: {kernel_size}")
            processed_img = apply_ideal_lowpass_filter(processed_img, kernel_size)
        elif filter_name == '高斯高通滤波器':
            ui.textBrowser.append(f"应用高斯高通滤波器，截止频率: {kernel_size}")
            processed_img = apply_gaussian_highpass_filter(processed_img, kernel_size)

    # 打印处理后的图像信息
    ui.textBrowser.append(f"处理后的图像尺寸: {processed_img.shape}")
    ui.textBrowser.append(f"处理后的图像像素值范围: {np.min(processed_img)} - {np.max(processed_img)}")
    
    return processed_img



# 初始化 EasyOCR 模型（支持中文和英文）
reader = easyocr.Reader(['ch_sim', 'en'])

# 图像预处理函数
def preprocess_image(image):
    """
    对图像进行预处理：灰度化、自适应直方图均衡化、局部自适应二值化。
    :param image: 原始 RGB 图像
    :return: 预处理后的图像
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值，避免过度二值化
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 锐化处理（可以增加细节）
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(adaptive_thresh, -1, kernel)

    # 可以选择加入去噪（中值滤波）
    denoised = cv2.medianBlur(sharpened, 3)

    # 转为三通道（适应 OCR）
    processed_image = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return processed_image
                            
# 点击提取文本的按钮事件
def click_pushButton_extract_text():
    ui.textBrowser.setPlainText(
        "提取文本（OCR）：OCR（光学字符识别）技术用于从图像中提取文本信息。"
        "通过使用深度学习模型或传统的图像处理方法，OCR 可以识别图像中的字符，并将其转换为可编辑的文本。"
        "该功能常用于扫描文档、数字化书籍、自动读取车牌号、票据等场景。"
        "OCR 系统的性能通常与图像质量、字符大小、字体类型等因素密切相关，处理效果受输入图像的影响较大。"
    )

    if ui.label.pixmap() is not None:
        # 从 label 中获取图片
        original_pixmap = ui.label.pixmap()
        original_image = original_pixmap.toImage()
        width, height = original_image.width(), original_image.height()

        # QImage 转 OpenCV 图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)  # 设置大小
        img_array = np.array(img_data).reshape((height, width, 4))  # BRGA 格式
        img_rgb = img_array[:, :, :3]  # 去掉 Alpha 通道，保留 RGB

        # 预处理图像
        processed_image = preprocess_image(img_rgb)

        # 使用 EasyOCR 提取文本
        results = reader.readtext(
            processed_image,
            detail=1,           # 是否返回详细信息
            text_threshold=0.8,  # 文本框内检测文字的置信度阈值
            low_text=0.4,        # 文字分割阈值
            link_threshold=0.4   # 文本连接阈值
        )

        # 显示提取的文本并在 label2 中绘制边框
        extracted_text = ""
        for (bbox, text, prob) in results:
            # 提取边框坐标
            (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]

            # 绘制绿色边框
            cv2.rectangle(processed_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # 拼接提取到的文本
            extracted_text += f"{text}\n"

        # 将结果显示在 textBrowser
        if extracted_text.strip():
            ui.textBrowser.setText(extracted_text)
        else:
            ui.textBrowser.setText("未检测到文本。")

        # 将处理后的图像显示在 label2
        height, width, channel = processed_image.shape
        step = channel * width
        q_img = QImage(processed_image.data, width, height, step, QImage.Format_RGB888)
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)
    else:
        QMessageBox.warning(ui, "警告", "未加载图像，请加载图像后再进行处理。")

def click_pushButton_26():
    text = """
    基于 YOLOv3 的人脸识别
    YOLOv3（You Only Look Once v3）是一个流行的目标检测算法，能够实时地检测图像中的目标物体，包括人脸。YOLOv3 是 YOLO 系列算法的第三个版本，具有更高的检测精度和速度。它通过将图像分成网格，并为每个网格预测边界框及其分类概率，从而快速而准确地检测目标物体。

    工作原理：
    YOLOv3 将图像划分为 SxS 的网格，每个网格预测多个边界框。每个边界框包含位置、大小以及与该框相关的目标类别的概率。在人脸检测任务中，YOLOv3 的输出包括人脸的坐标、大小以及置信度。

    优势：
    - **速度**：YOLOv3 是一个高效的检测模型，适用于实时应用。
    - **准确性**：通过多层次的卷积神经网络（CNN），YOLOv3 能够对不同大小的目标进行精确检测。
    - **端到端训练**：YOLOv3 是一个端到端的模型，不需要区域提议步骤，因此非常高效。

    使用 YOLOv3 进行人脸识别的步骤：
    1. 通过预训练模型加载 YOLOv3 权重。
    2. 对输入图像进行预处理（例如调整尺寸、归一化）。
    3. 使用 YOLOv3 模型进行预测，检测图像中的人脸。
    4. 在图像中标出检测到的人脸区域。

    相关参数：
    - **置信度阈值**：用于控制检测结果的精度，通常设置为 0.5 以上。
    - **NMS（非极大值抑制）**：用于去除冗余的边界框，只保留最优的检测结果。

    YOLOv3 广泛应用于实时物体检测和人脸识别任务，具有较好的性能表现。
    """
    ui.textBrowser.setPlainText(text)  # 显示文字到 textBrowser 中
    # 检查label中的pixmap是否存在
    if ui.label.pixmap() is not None:
        # 获取原图像（QPixmap -> QImage）
        original_image = ui.label.pixmap().toImage()
        width, height = original_image.width(), original_image.height()

        # 将 QImage 转换为 OpenCV 格式的图像
        img_data = original_image.bits()
        img_data.setsize(height * width * 4)
        img_array = np.array(img_data).reshape((height, width, 4))  # BGRA 格式
        img_rgb = img_array[:, :, :3]  # 只保留 RGB 通道

        # 确保 img_rgb 的数据类型是 uint8
        img_rgb = img_rgb.astype(np.uint8)

        # 使用YOLO进行检测
        results = model(img_rgb)  # 返回的是一个列表，包含一个 Results 对象

        # 获取检测结果的第一个元素（YOLOv5返回一个包含检测结果的列表）
        result = results[0]

        # 获取YOLO检测到的边界框（boxes）和对应的类别（classes）
        boxes = result.boxes  # 获取边界框
        classes = boxes.cls.numpy()  # 获取类别
        confidences = boxes.conf.numpy()  # 获取置信度
        xywh = boxes.xywh.numpy()  # 获取归一化坐标

        # 设置置信度阈值，过滤掉低于阈值的框
        confidence_threshold = 0.3  # 调整为较低的阈值

        # 过滤出人脸类别的框（类ID 0是人脸，通常可以根据类别ID来筛选）
        for i, cls in enumerate(classes):
            if confidences[i] > confidence_threshold and cls == 0:  # 只选择置信度大于阈值的框
                # 获取归一化坐标，转换为像素坐标
                x_center, y_center, w, h = xywh[i]

                # 计算边界框的左上角和右下角坐标
                x1 = int(x_center - w / 2) 
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # 打印框的坐标，确认是否正确
                print(f"Drawing rectangle: ({x1}, {y1}), ({x2}, {y2})")

                # 绘制绿色矩形框
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 将处理后的图像转换为QImage以便显示
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # 转为RGB格式
        height, width, channel = img_rgb.shape
        step = channel * width
        q_img = QImage(img_rgb.data, width, height, step, QImage.Format_RGB888)

        # 显示处理后的图像
        ui.label_2.setPixmap(QPixmap.fromImage(q_img))
        ui.label_2.setScaledContents(True)
# ==================== 摄像头相关功能 ====================

class CameraApp:
    def __init__(self):
        # 摄像头对象和定时器
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def click_pushButton_2(self):
        """打开/关闭摄像头"""
        if not self.capture:
            # 打开摄像头
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("无法打开摄像头")
                self.capture = None
                return
            # 开启定时器，每 30ms 刷新一帧
            self.timer.start(30)
            ui.pushButton_2.setText("关闭摄像头")
        else:
            # 关闭摄像头
            self.timer.stop()
            self.capture.release()
            self.capture = None
            ui.label.clear()  # 清空 QLabel 显示
            ui.pushButton_2.setText("打开摄像头")

    def update_frame(self):
        """实时更新摄像头图像并显示"""
        ret, frame = self.capture.read()
        if ret:
            # 转换图像为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            step = channel * width
            # 创建 QImage
            q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            ui.label.setPixmap(QPixmap.fromImage(q_img))

# ==================== 程序主入口 ====================

if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = dip.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setStyleSheet("""
QWidget {
    font-family: "华文行楷";
    font-size: 13pt;  /* 假设原本的字号为11pt，增加1 */
}                             
QPushButton{
	border-style: solid;
	border-color: #050a0e;
	border-width: 1px;
	border-radius: 5px;
	color: #d3dae3;
	padding: 2px;
	background-color: #100E19;
}
QPushButton::default{
	border-style: solid;
	border-color: #050a0e;
	border-width: 1px;
	border-radius: 5px;
	color: #FFFFFF;
	padding: 2px;
	background-color: #151a1e;
}
QPushButton:hover{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:1, stop:0 #C0DB50, stop:0.4 #C0DB50, stop:0.5 #100E19, stop:1 #100E19);
    border-bottom-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:1, stop:0 #100E19, stop:0.5 #100E19, stop:0.6 #C0DB50, stop:1 #C0DB50);
    border-left-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #C0DB50, stop:0.3 #C0DB50, stop:0.7 #100E19, stop:1 #100E19);
    border-right-color: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #C0DB50, stop:0.3 #C0DB50, stop:0.7 #100E19, stop:1 #100E19);
	border-width: 2px;
    border-radius: 1px;
	color: #d3dae3;
	padding: 2px;
}
QPushButton:pressed{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:1, stop:0 #d33af1, stop:0.4 #d33af1, stop:0.5 #100E19, stop:1 #100E19);
    border-bottom-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:1, stop:0 #100E19, stop:0.5 #100E19, stop:0.6 #d33af1, stop:1 #d33af1);
    border-left-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #d33af1, stop:0.3 #d33af1, stop:0.7 #100E19, stop:1 #100E19);
    border-right-color: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d33af1, stop:0.3 #d33af1, stop:0.7 #100E19, stop:1 #100E19);
	border-width: 2px;
    border-radius: 1px;
	color: #d3dae3;
	padding: 2px;
}
QMainWindow {
	background-color:#151a1e;
}
QCalendar {
	background-color: #151a1e;
}
QTextEdit {
	border-width: 1px;
	border-style: solid;
	border-color: #4fa08b;
	background-color: #222b2e;
	color: #d3dae3;
}
QPlainTextEdit {
	border-width: 1px;
	border-style: solid;
	border-color: #4fa08b;
	background-color: #222b2e;
	color: #d3dae3;
}
QLineEdit {
	border-width: 1px;
	border-style: solid;
	border-color: #4fa08b;
	background-color: #222b2e;
	color: #d3dae3;
}
QLabel {
	color:rgb(82, 85, 88);
}
QLCDNumber {
	color: #4d9b87;
}
QProgressBar {
	text-align: center;
	color: #d3dae3;
	border-radius: 10px;
	border-color: transparent;
	border-style: solid;
	background-color: #52595d;
}
QProgressBar::chunk {
	background-color: #214037	;
	border-radius: 10px;
}
QMenuBar {
	background-color: #151a1e;
}
QMenuBar::item {
	color: #d3dae3;
  	spacing: 3px;
  	padding: 1px 4px;
	background-color: #151a1e;
}

QMenuBar::item:selected {
  	background-color: #252a2e;
	color: #FFFFFF;
}
QMenu {
	background-color: #151a1e;
}
QMenu::item:selected {
	background-color: #252a2e;
	color: #FFFFFF;
}
QMenu::item {
	color: #d3dae3;
	background-color: #151a1e;
}
QTabWidget {
	color:rgb(0,0,0);
	background-color:#000000;
}
QTabWidget::pane {
		border-color: #050a0e;
		background-color: #1e282c;
		border-style: solid;
		border-width: 1px;
    	border-bottom-left-radius: 4px;
		border-bottom-right-radius: 4px;
}
QTabBar::tab:first {
	border-style: solid;
	border-left-width:1px;
	border-right-width:0px;
	border-top-width:1px;
	border-bottom-width:0px;
	border-top-color: #050a0e;
	border-left-color: #050a0e;
	border-bottom-color: #050a0e;
	border-top-left-radius: 4px;
	color: #d3dae3;
	padding: 3px;
	margin-left:0px;
	background-color: #151a1e;
}
QTabBar::tab:last {
	border-style: solid;
	border-top-width:1px;
	border-left-width:1px;
	border-right-width:1px;
	border-bottom-width:0px;
	border-color: #050a0e;
	border-top-right-radius: 4px;
	color: #d3dae3;
	padding: 3px;
	margin-left:0px;
	background-color: #151a1e;
}
QTabBar::tab {
	border-style: solid;
	border-top-width:1px;
	border-bottom-width:0px;
	border-left-width:1px;
	border-top-color: #050a0e;
	border-left-color: #050a0e;
	border-bottom-color: #050a0e;
	color: #d3dae3;
	padding: 3px;
	margin-left:0px;
	background-color: #151a1e;
}
QTabBar::tab:selected, QTabBar::tab:last:selected, QTabBar::tab:hover {
  	border-style: solid;
  	border-left-width:1px;
	border-bottom-width:0px;
	border-right-color: transparent;
	border-top-color: #050a0e;
	border-left-color: #050a0e;
	border-bottom-color: #050a0e;
	color: #FFFFFF;
	padding: 3px;
	margin-left:0px;
	background-color: #1e282c;
}

QTabBar::tab:selected, QTabBar::tab:first:selected, QTabBar::tab:hover {
  	border-style: solid;
  	border-left-width:1px;
  	border-bottom-width:0px;
  	border-top-width:1px;
	border-right-color: transparent;
	border-top-color: #050a0e;
	border-left-color: #050a0e;
	border-bottom-color: #050a0e;
	color: #FFFFFF;
	padding: 3px;
	margin-left:0px;
	background-color: #1e282c;
}

QCheckBox {
	color: #d3dae3;
	padding: 2px;
}
QCheckBox:disabled {
	color: #808086;
	padding: 2px;
}

QCheckBox:hover {
	border-radius:4px;
	border-style:solid;
	padding-left: 1px;
	padding-right: 1px;
	padding-bottom: 1px;
	padding-top: 1px;
	border-width:1px;
	border-color: transparent;
}
QCheckBox::indicator:checked {

	height: 10px;
	width: 10px;
	border-style:solid;
	border-width: 1px;
	border-color: #4fa08b;
	color: #000000;
	background-color: qradialgradient(cx:0.4, cy:0.4, radius: 1.5,fx:0, fy:0, stop:0 #1e282c, stop:0.3 #1e282c, stop:0.4 #4fa08b, stop:0.5 #1e282c, stop:1 #1e282c);
}
QCheckBox::indicator:unchecked {

	height: 10px;
	width: 10px;
	border-style:solid;
	border-width: 1px;
	border-color: #4fa08b;
	color: #000000;
}

QStatusBar {
	color:#027f7f;
}

QFontComboBox {
	color: #d3dae3;
	background-color: #222b2e;
	border-width: 1px;
	border-style: solid;
	border-color: #4fa08b;
}
QComboBox {
	color: #d3dae3;
	background-color: #222b2e;
	border-width: 1px;
	border-style: solid;
	border-color: #4fa08b;
}

QDial {
	background: #16a085;
}

QToolBox {
	color: #a9b7c6;
	background-color: #222b2e;
}
QToolBox::tab {
	color: #a9b7c6;
	background-color:#222b2e;
}
QToolBox::tab:selected {
	color: #FFFFFF;
	background-color:#222b2e;
}
QScrollArea {
	color: #FFFFFF;
	background-color:#222b2e;
}
QSlider::groove:horizontal {
	height: 5px;
	background-color: #52595d;
}
QSlider::groove:vertical {
	width: 5px;
	background-color: #52595d;
}
QSlider::handle:horizontal {
	background: #1a2224;
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	width: 12px;
	margin: -5px 0;
	border-radius: 7px;
}
QSlider::handle:vertical {
	background: #1a2224;
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	height: 12px;
	margin: 0 -5px;
	border-radius: 7px;
}
QSlider::add-page:horizontal {
    background: #52595d;
}
QSlider::add-page:vertical {
    background: #52595d;
}
QSlider::sub-page:horizontal {
    background-color: #15433a;
}
QSlider::sub-page:vertical {
    background-color: #15433a;
}
QScrollBar:horizontal {
	max-height: 10px;
	border: 1px transparent grey;
	margin: 0px 20px 0px 20px;
	background: transparent;
}
QScrollBar:vertical {
	max-width: 10px;
	border: 1px transparent grey;
	margin: 20px 0px 20px 0px;
	background: transparent;
}
QScrollBar::handle:horizontal {
	background: #52595d;
	border-style: transparent;
	border-radius: 4px;
	min-width: 25px;
}
QScrollBar::handle:horizontal:hover {
	background: #58a492;
	border-style: transparent;
	border-radius: 4px;
	min-width: 25px;
}
QScrollBar::handle:vertical {
	background: #52595d;
	border-style: transparent;
	border-radius: 4px;
	min-height: 25px;
}
QScrollBar::handle:vertical:hover {
	background: #58a492;
	border-style: transparent;
	border-radius: 4px;
	min-height: 25px;
}
QScrollBar::add-line:horizontal {
   border: 2px transparent grey;
   border-top-right-radius: 4px;
   border-bottom-right-radius: 4px;
   background: #15433a;
   width: 20px;
   subcontrol-position: right;
   subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal:pressed {
   border: 2px transparent grey;
   border-top-right-radius: 4px;
   border-bottom-right-radius: 4px;
   background: rgb(181,181,181);
   width: 20px;
   subcontrol-position: right;
   subcontrol-origin: margin;
}
QScrollBar::add-line:vertical {
   border: 2px transparent grey;
   border-bottom-left-radius: 4px;
   border-bottom-right-radius: 4px;
   background: #15433a;
   height: 20px;
   subcontrol-position: bottom;
   subcontrol-origin: margin;
}
QScrollBar::add-line:vertical:pressed {
   border: 2px transparent grey;
   border-bottom-left-radius: 4px;
   border-bottom-right-radius: 4px;
   background: rgb(181,181,181);
   height: 20px;
   subcontrol-position: bottom;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal {
   border: 2px transparent grey;
   border-top-left-radius: 4px;
   border-bottom-left-radius: 4px;
   background: #15433a;
   width: 20px;
   subcontrol-position: left;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal:pressed {
   border: 2px transparent grey;
   border-top-left-radius: 4px;
   border-bottom-left-radius: 4px;
   background: rgb(181,181,181);
   width: 20px;
   subcontrol-position: left;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical {
   border: 2px transparent grey;
   border-top-left-radius: 4px;
   border-top-right-radius: 4px;
   background: #15433a;
   height: 20px;
   subcontrol-position: top;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical:pressed {
   border: 2px transparent grey;
   border-top-left-radius: 4px;
   border-top-right-radius: 4px;
   background: rgb(181,181,181);
   height: 20px;
   subcontrol-position: top;
   subcontrol-origin: margin;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
   background: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
   background: none;
}
    """)
    # 设置 QLabel 外观
    ui.label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label.setText("图片将在这里显示")
    ui.label.setAlignment(Qt.AlignCenter)
    
    ui.label_2.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label_2.setText("处理后的图像显示在这里")
    ui.label_3.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label_3.setAlignment(Qt.AlignCenter)
    ui.label_4.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label_4.setAlignment(Qt.AlignCenter)
    ui.label_5.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label_5.setAlignment(Qt.AlignCenter)
    ui.label_6.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label_6.setAlignment(Qt.AlignCenter)
    ui.label_7.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label_7.setAlignment(Qt.AlignCenter)
    ui.label_8.setStyleSheet("background-color: lightgray; border: 1px solid black;")
    ui.label_8.setAlignment(Qt.AlignCenter)
    # 设置滑块初始值和范围
    ui.horizontalSlider.setMinimum(1)  # 滤波核最小值
    ui.horizontalSlider.setMaximum(100)  # 滤波核最大值
    ui.horizontalSlider.setValue(10)  # 初始值
    ui.label_slider_value.setText("核大小: 10x10")
    ui.label_slider_value_2.setText("截止频率: 10")
    ui.label_slider_value_3.setText("元素大小: 10")
    ui.label_slider_value_4.setText("阈值: 10")
    setup_filter_combobox()  # 设置滤波器选择下拉菜单
    # 创建 CameraApp 实例
    camera_app = CameraApp()

    # 连接按钮和滑块事件
    ui.horizontalSlider.valueChanged.connect(update_slider_label)  # 滑块值变化
    ui.horizontalSlider.valueChanged.connect(update_slider_label_2)  # 滑块值变化
    ui.horizontalSlider.valueChanged.connect(update_slider_label_3)  # 滑块值变化
    ui.horizontalSlider.valueChanged.connect(update_slider_label_4)  # 滑块值变化
    ui.pushButton.clicked.connect(click_pushButton)  # 选择图片
    ui.pushButton_2.clicked.connect(camera_app.click_pushButton_2)  # 打开/关闭摄像头
    ui.pushButton_3.clicked.connect(click_pushButton_3)
    
    ui.pushButton_4.clicked.connect(click_pushButton_4)  # 应用均值滤波
    ui.pushButton_5.clicked.connect(click_pushButton_5)  # 应用高斯滤波   
    ui.pushButton_6.clicked.connect(click_pushButton_6)  # 应用 Sobel 边缘检测并显示处理后的图像
    ui.pushButton_7.clicked.connect(click_pushButton_7)  # 应用拉普拉斯滤波并显示处理后的图像
    ui.pushButton_15.clicked.connect(click_pushButton_15)  # 应用拉普拉斯滤波并显示处理后的图像
    ui.pushButton_8.clicked.connect(click_pushButton_8)  # 绑定傅里叶变换按钮
    ui.pushButton_9.clicked.connect(click_pushButton_9)  # 绑定理想低通变换按钮
    ui.pushButton_10.clicked.connect(click_pushButton_10)  # 绑定高斯高通变换按钮
    ui.pushButton_11.clicked.connect(click_pushButton_11)  # 应用腐蚀操作并显示处理后的图像
    ui.pushButton_12.clicked.connect(click_pushButton_12)  # 应用膨胀操作并显示处理后的图像
    ui.pushButton_13.clicked.connect(click_pushButton_13)  # 应用开运算操作并显示处理后的图像
    ui.pushButton_14.clicked.connect(click_pushButton_14)  # 应用闭运算操作并显示处理后的图像
    ui.pushButton_16.clicked.connect(click_pushButton_16)  # 应用形态学梯度操作并显示处理后的图像
    ui.pushButton_17.clicked.connect(click_pushButton_17)  # 应用全局阈值分割
    ui.pushButton_18.clicked.connect(click_pushButton_18)  # 应用自适应阈值分割
    ui.pushButton_19.clicked.connect(click_pushButton_19)  # 应用 Otsu 阈值分割
    ui.pushButton_20.clicked.connect(click_pushButton_20)  # 应用 Canny 边缘检测
    ui.pushButton_21.clicked.connect(click_pushButton_21)  # 应用 DeepLabV3+ 图像分割
    ui.pushButton_22.clicked.connect(click_pushButton_22)  # 应用 MASK R-CNN 图像分割
    ui.pushButton_23.clicked.connect(batch_process_images)  # 批量处理图片
    ui.pushButton_24.clicked.connect(process_single_image)  #流水线
    ui.pushButton_25.clicked.connect(click_pushButton_extract_text)  #获取文本
    ui.pushButton_26.clicked.connect(click_pushButton_26)  #YOLOV5人脸检测
    # 连接每个checkbox的状态变化事件，传递checkbox名称及勾选状态
    ui.checkBox.stateChanged.connect(lambda state: on_checkbox_state_changed('均值滤波', state == Qt.Checked))
    ui.checkBox_2.stateChanged.connect(lambda state: on_checkbox_state_changed('高斯滤波', state == Qt.Checked))
    ui.checkBox_3.stateChanged.connect(lambda state: on_checkbox_state_changed('Sobel边缘检测', state == Qt.Checked))
    ui.checkBox_4.stateChanged.connect(lambda state: on_checkbox_state_changed('拉普拉斯滤波', state == Qt.Checked))
    ui.checkBox_5.stateChanged.connect(lambda state: on_checkbox_state_changed('中值滤波', state == Qt.Checked))
    ui.checkBox_6.stateChanged.connect(lambda state: on_checkbox_state_changed('腐蚀', state == Qt.Checked))
    ui.checkBox_7.stateChanged.connect(lambda state: on_checkbox_state_changed('膨胀', state == Qt.Checked))
    ui.checkBox_8.stateChanged.connect(lambda state: on_checkbox_state_changed('开运算', state == Qt.Checked))
    ui.checkBox_9.stateChanged.connect(lambda state: on_checkbox_state_changed('闭运算', state == Qt.Checked))
    ui.checkBox_10.stateChanged.connect(lambda state: on_checkbox_state_changed('梯度运算', state == Qt.Checked))
    ui.checkBox_11.stateChanged.connect(lambda state: on_checkbox_state_changed('理想低通滤波器', state == Qt.Checked))
    ui.checkBox_12.stateChanged.connect(lambda state: on_checkbox_state_changed('高斯高通滤波器', state == Qt.Checked))
    
    # 显示主窗口并进入事件循环
    MainWindow.show()
    sys.exit(app.exec_())
