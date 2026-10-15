import numpy as np
from PIL import Image
import pandas as pd
import cv2
import os

"""
def overlay_images_with_transparency(background_array, overlay_array, alpha=0.5, black_threshold=10, output_path=None):

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
"""

def overlay_images_with_transparency(background_array, overlay_array, alpha=0.5, black_threshold=10, output_path=None):

    if background_array.shape[-1] != 3:
        raise ValueError("background_array 必须是 (H, W, 3) 的 RGB 图像！")

    if len(overlay_array.shape) != 2:
        raise ValueError("overlay_array 必须是 (H, W) 的灰度图！")

    # 创建红色叠加层
    red_overlay = np.zeros_like(background_array, dtype=np.float32)
    red_overlay[..., 2] = 255  # 红色通道

    # 生成 Alpha 通道（透明度）
    alpha_mask = np.where(overlay_array > black_threshold, alpha, 0).astype(np.float32)

    # 将背景图和叠加层按透明度混合
    combined_rgb = (1 - alpha_mask[..., None]) * background_array + alpha_mask[..., None] * red_overlay

    # 转换为 uint8 格式
    combined_rgb = np.clip(combined_rgb, 0, 255).astype(np.uint8)
    combined_rgb = cv2.cvtColor(combined_rgb, cv2.COLOR_BGR2RGB)
    # 保存结果
    if output_path:
        Image.fromarray(combined_rgb).save(output_path)
        print(f"overlay image saved")

    return combined_rgb


def overlay_images(background, overlay, alpha=0.3):
    """
    将 overlay 图像叠加到 background 图像上，叠加透明度为 alpha。

    参数:
        background (numpy.ndarray): 背景图像，形状为 (H, W, 3)，RGB 格式。
        overlay (numpy.ndarray): 叠加图像，形状为 (H, W, 3)，RGB 格式。
        alpha (float): 叠加图像的透明度，范围 [0, 1]。

    返回:
        numpy.ndarray: 融合后的图像，形状为 (H, W, 3)。
    """
    # 确保两张图像的尺寸一致
    if background.shape != overlay.shape:
        raise ValueError("两张图片的尺寸不一致！请调整到相同尺寸。")

    # 融合两张图像
    result = alpha * overlay + (1 - alpha) * background

    # 确保像素值在 [0, 255] 范围内，并转换为 uint8 类型
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def hex_to_bgr(hex_color):
    """
    将十六进制颜色值转换为OpenCV使用的BGR格式。

    参数:
        hex_color (str): 16进制颜色值（以#开头，例如 "#0771af"）。

    返回:
        tuple: 转换后的BGR格式颜色元组，例如 (175, 113, 7)。
    """
    # 去掉#并分割成RGB部分
    r_hex = hex_color[1:3]
    g_hex = hex_color[3:5]
    b_hex = hex_color[5:7]

    # 将十六进制转换为十进制
    r = int(r_hex, 16)
    g = int(g_hex, 16)
    b = int(b_hex, 16)

    # 返回BGR格式
    return (b, g, r)



def brain_region_contour_overlay(resized_transformed_mask_uint8, raw_image, bgr_color):

    # 阈值分割（Otsu 二值化）
    _, binary_img = cv2.threshold(resized_transformed_mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 去噪（形态学操作）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    # 找到连通区域
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像（和原图同大小）用于绘制绿色线条
    green_lines = np.zeros((resized_transformed_mask_uint8.shape[0], resized_transformed_mask_uint8.shape[1], 3), dtype=np.uint8)

    # 在空白图像上绘制绿色线条
    for contour in contours:
        cv2.drawContours(green_lines, [contour], -1, bgr_color, thickness=4)

    # 确保目标图像大小与绿色线条图层一致（如有必要，可以调整大小）
    target_img = cv2.resize(raw_image, (resized_transformed_mask_uint8.shape[1], resized_transformed_mask_uint8.shape[0]))

    # 将绿色线条叠加到目标图像
    result_img = cv2.addWeighted(target_img, 1.0, green_lines, 1.0, 0)
    
    return result_img





def total_brain_contour_overlay(resized_transformed_annotation, raw_image, bgr_color):

    # 将注释图像转换为 HSV 色彩空间
    hsv_img = cv2.cvtColor(resized_transformed_annotation, cv2.COLOR_RGB2HSV)

    # 创建一个空白图像用于绘制绿色轮廓
    green_lines = np.zeros_like(resized_transformed_annotation)

    # 定义颜色范围列表
    color_ranges = [
        ((0, 10, 50), (10, 79, 255)),    # 红色 (低范围)
        ((0, 80, 50), (10, 110, 255)),
        ((0, 111, 50), (10, 125, 255)),
        ((0, 126, 50), (10, 255, 255)),
        ((160, 50, 50), (179, 255, 255)),  # 红色 (高范围)
        ((160, 10, 50), (179, 49, 255)),
        ((10, 151, 50), (25, 255, 255)),   # 橙色
        ((10, 10, 50), (25, 150, 255)),
        ((25, 50, 50), (35, 255, 255)),    # 黄色
        ((25, 10, 50), (35, 49, 255)),
        ((35, 121, 50), (85, 255, 255)),   # 绿色
        ((35, 10, 50), (85, 120, 255)),
        ((85, 50, 50), (100, 255, 255)),   # 青色
        ((85, 10, 50), (100, 49, 255)),
        ((100, 130, 50), (130, 255, 255)), # 蓝色
        ((100, 110, 50), (130, 129, 255)),
        ((100, 10, 50), (130, 109, 255)),
        ((130, 50, 50), (160, 255, 255)),  # 紫色
        ((130, 10, 50), (160, 49, 255)),   # 自定义颜色范围
    ]

    # 遍历所有颜色范围，生成轮廓
    for lower, upper in color_ranges:
        # 根据颜色范围创建掩码
        mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
    
        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # 绘制轮廓到绿色线条图像上
        for contour in contours:
            cv2.drawContours(green_lines, [contour], -1, bgr_color, thickness=2)

    # 将绿色线条叠加到原始图像
    result_img = cv2.addWeighted(raw_image, 1.0, green_lines, 1.0, 0)
    
    return result_img




def colorize(channels, zslice, term_set):
    sshape = zslice.shape
    colorized = np.zeros((sshape[0], sshape[1], 3), dtype=np.uint8)

    for i, c in enumerate(['red', 'green', 'blue']):
        temp = np.zeros((sshape[0], sshape[1]), dtype=np.uint8)

        # 自动判断 channels[c] 是 DataFrame 还是 Series
        if isinstance(channels[c], pd.DataFrame):
            temp.flat[:] = channels[c].loc[zslice.flat[:], f'{term_set}_color']
        else:
            temp.flat[:] = channels[c].loc[zslice.flat[:]]

        colorized[:, :, i] = temp

    return colorized

