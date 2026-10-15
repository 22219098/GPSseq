import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2



##Affine model
def Affine_label():
    """
    创建一个模型，用于处理输入的灰度图像并进行二值化操作。
    """
    # 定义输入，形状为 (None, 256, 256, 1)
    input_image = tf.keras.Input(shape=(256, 256, 1), dtype=tf.float32)

    # 使用 TensorFlow 的操作进行二值化
    binary_map = tf.keras.layers.Lambda(lambda x: tf.where(x > 15, 1.0, 0.0))(input_image)

    # 创建模型，输出为原始输入图像和二值化后的图像
    model = tf.keras.Model(inputs=input_image, outputs=[input_image, binary_map])

    return model


def process_with_affine(input_data):
    """
    用 Affine_label 模型直接处理输入数据。
    """
    model = Affine_label()  # 每次调用时创建新的模型实例
    return model(input_data)
    

def image_affine_with_ConvNext(step1_prediction, ConvNext_template, output_size=(256, 256), batch_size = 1):

    label_maps = []
    template_out = []

    # Process step1_prediction directly
    if len(step1_prediction.shape) == 3 and step1_prediction.shape[2] > 1:
        # Convert to grayscale if the input is a color image
        step1_prediction = cv2.cvtColor(step1_prediction, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(step1_prediction, output_size, interpolation=cv2.INTER_AREA)
    
    # Process ConvNext_template directly
    if len(ConvNext_template.shape) == 3 and ConvNext_template.shape[2] > 1:
        # Convert to grayscale if the input is a color image
        ConvNext_template = cv2.cvtColor(ConvNext_template, cv2.COLOR_BGR2GRAY)
    template_resized = cv2.resize(ConvNext_template, output_size, interpolation=cv2.INTER_AREA)   

    # Append the processed image to the label_maps list
    label_maps.append(img_resized)
    template_out.append(template_resized)

    in_shape = label_maps[0].shape
    
    # Expand dims to add the channel dimension (if not already present)
    label_maps = np.expand_dims(label_maps, axis=-1)
    labels_out = np.expand_dims(template_out, axis=-1)
    return labels_out, label_maps, in_shape
    
    
def overlay_images_with_transparency(background_array, overlay_array, alpha=0.5, black_threshold=10, output_path=None):
    """
    将灰度图叠加到 RGB 图上，去掉灰度图中的接近黑色的部分。

    参数：
        background_array (numpy.ndarray): 背景 RGB 图像，形状为 (H, W, 3)。
        overlay_array (numpy.ndarray): 灰度图，形状为 (H, W)。
        alpha (float): 叠加图的透明度，范围 [0, 1]。
        black_threshold (int): 将灰度值小于此阈值的像素视为黑色。
        output_path (str): 合成图像的保存路径，可选。

    返回：
        numpy.ndarray: 合成后的 RGB 图像。
    """
    # 确保背景图是 RGB 格式
    if background_array.shape[-1] != 3:
        raise ValueError("background_array 必须是 (H, W, 3) 的 RGB 图像！")

    # 确保 overlay_array 是灰度图
    if len(overlay_array.shape) != 2:
        raise ValueError("overlay_array 必须是 (H, W) 的灰度图！")

    # 将 overlay_array 扩展为 RGBA 格式 (H, W, 4)
    overlay_rgba = np.zeros((overlay_array.shape[0], overlay_array.shape[1], 4), dtype=np.uint8)
    overlay_rgba[..., :3] = np.stack([overlay_array] * 3, axis=-1)  # 复制灰度值到 R, G, B 通道

    # 设置透明度：只有大于阈值的像素才应用透明度
    overlay_rgba[..., 3] = np.where(overlay_array > black_threshold, int(alpha * 255), 0)

    # 将背景图转换为 RGBA 格式
    background_rgba = np.concatenate([background_array, np.full((background_array.shape[0], background_array.shape[1], 1), 255, dtype=np.uint8)], axis=-1)

    # 叠加图像
    combined_rgba = background_rgba.copy()
    mask = overlay_rgba[..., 3] > 0  # 找到非透明区域
    combined_rgba[mask] = overlay_rgba[mask]

    # 转换为 RGB 格式
    combined_rgb = combined_rgba[..., :3]
    combined_rgb = cv2.cvtColor(combined_rgb, cv2.COLOR_BGR2RGB)

    # 保存结果（如果需要）
    if output_path:
        Image.fromarray(combined_rgb).save(output_path)
        print(f"overlay image saved")

    return combined_rgb



