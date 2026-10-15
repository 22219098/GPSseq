import tensorflow as tf
from keras import ops
import os
import cv2
import numpy as np

input_img_size = (256, 256, 3)

#################adjust3###########################

def check_image_group(raw_image_path):

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    if os.path.isdir(raw_image_path):  # 检测是否为文件夹
        return True
    elif os.path.isfile(raw_image_path) and os.path.splitext(raw_image_path)[1].lower() in valid_extensions:
        return False
    else:
        raise ValueError("No image or folder found!")

def parse_brain_regions(brain_region):
    if brain_region is None:
        return []

    if isinstance(brain_region, str):
        return [x.strip() for x in brain_region.split(",") if x.strip()]

    if isinstance(brain_region, (list, tuple, set)):
        return [str(x).strip() for x in brain_region if str(x).strip()]

    raise TypeError("brain_region must be str、list、tuple、set or None")


def apply_clahe_opencv(img):

    # 检查 img 是否为 NumPy 数组，如果不是，将其转换为 NumPy 数组
    if not isinstance(img, np.ndarray):
        img = img.numpy()  # 将 TensorFlow 张量转换为 NumPy 数组
    # 确保输入图像是 uint8 类型
    img = img.astype(np.uint8)
    
    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用 CLAHE
    img = clahe.apply(img)
    # 将 2D 图像扩展为 3D (256, 256, 1)
    img = np.expand_dims(img, axis=-1)
    return img

def apply_clahe(img):

    # 确保输入是 [0, 255] 的 uint8 类型
    img = tf.cast(img, tf.uint8)
    # 调用 OpenCV 的 CLAHE 处理（通过 tf.py_function）
    enhanced_img = tf.py_function(func=apply_clahe_opencv, inp=[img], Tout=tf.uint8)
    enhanced_img = tf.ensure_shape(enhanced_img, (256, 256, 1))
    enhanced_img = tf.cast(enhanced_img, tf.float32) 
    return enhanced_img

def normalize_raw_img(img):
    # 1. 转灰度并计算平均亮度
    img = tf.image.rgb_to_grayscale(img)  # (H, W, 1), [0–255]
    avg = tf.reduce_mean(img)

    # --- 逻辑 A：图像处理模式 (保留第一段代码的 3 种模式) ---
    # 定义三个处理分支
    def low_fn():
        # 低亮度 (<20)：+80 then CLAHE
        bright = tf.clip_by_value(img + 80.0, 0.0, 255.0)
        return apply_clahe(bright)

    def high_fn():
        # 高亮度 (>240)：-10 then CLAHE
        dark = tf.clip_by_value(img - 10.0, 0.0, 255.0)
        return apply_clahe(dark)

    def mid_fn():
        # 中间亮度 (20-240)：直接 CLAHE
        return apply_clahe(img)

    # 根据 avg < 20 或 avg > 240 选择图像处理方式
    processed_img = tf.cond(
        avg < 20.0,
        low_fn,
        lambda: tf.cond(avg > 240.0, high_fn, mid_fn)
    )

    # --- 逻辑 B：场景标签判定 (引入第二段代码的功能) ---
    # 根据 avg > 70 判定是明场还是暗场
    # 使用 tf.constant 返回字符串 tensor
    field = tf.cond(
        avg > 70.0,
        lambda: tf.constant("bright_field"), # 亮度 > 70
        lambda: tf.constant("dark_field")    # 亮度 <= 70
    )

    # 3. 归一化 (保留第一段代码的 [-1, 1] 归一化逻辑，如需 [0, 1] 请修改此处)
    processed_img = processed_img / 255.0
    processed_img = processed_img * 2.0 - 1.0

    # 同时返回 处理后的图像 和 场景标签
    return processed_img, field

def preprocess_raw_image(img_input):
    # 读取和解码
    if isinstance(img_input, str):  # 如果是文件路径
        img = tf.io.read_file(img_input)
        img = tf.image.decode_image(img, channels=3)
    elif isinstance(img_input, np.ndarray):  # 如果是 NumPy 图像
        if img_input.dtype != np.uint8:
            img_input = img_input.astype(np.uint8)
        img = tf.convert_to_tensor(img_input)
    else:
        raise ValueError("输入必须是图像路径字符串或 numpy.ndarray 图像。")
    # Resize 需要根据你定义的 input_img_size，这里假设你有定义
    # img = tf.image.resize(img, [input_img_size[0], input_img_size[1]]) 
    img = tf.image.resize(img, [256, 256]) # 举例: 假设大小为 256x256
    
    # 获取图像和标签
    img, field = normalize_raw_img(img)
    
    return img, field

###############################################

def normalize_img(img):
    # 将图片灰度化
    img = tf.image.rgb_to_grayscale(img)  # 转为灰度图，结果为 (height, width, 1)
    # 将灰度图像素值归一化到 [-1, 1]
    img = img / 255.0
    img = img * 2.0 - 1.0
    return img


def preprocess_train_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, orig_img_size)
    img = tf.image.random_crop(img, size=input_img_size)
    return normalize_img(img)

def preprocess_template_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    return normalize_img(img)

def preprocess_single_image(img_path):
    """预处理单张图片"""
    # 读取图片并解码为张量
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize到256x256
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    # 归一化到[-1, 1]
    img = normalize_raw_img(img)
    return img

def scale(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# 修改 scale 函数，支持直接输入 NumPy 数组
def scale_from_array(img: np.ndarray) -> np.ndarray:
    # 确保输入是 NumPy 数组
    if isinstance(img, tf.Tensor):  # 如果是 TensorFlow 张量，先转为 NumPy
        img = img.numpy()

    # Resize 到 (256, 256)
    img = cv2.resize(img, (256, 256))

    # 灰度化处理（如果是多通道图像）
    if len(img.shape) == 3 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def load_images(image_folder):
    """Load images from a folder and return data and labels."""
    images = os.listdir(image_folder)
    data = []
    labels = []

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = scale(img_path)
        data.append(img)
        labels.append(os.path.splitext(image)[0])
    
    data = np.array(data)
    return data, labels

def process_images_with_template(input_folder, output_size=(256, 256)):
    """
    Process PNG images to label map format using OpenCV (grayscale, resized, and as NumPy arrays),
    and return corresponding labels (as integers) and labels_out.

    Parameters:
        input_folder (str): Path to the folder containing PNG images.
        output_size (tuple): Desired size for the output images (default: 256x256).

    Returns:
        list: A list of NumPy arrays representing processed label maps.
        np.array: Labels (integers) corresponding to the processed label maps.
        dict: labels_out mapping image names (without extensions) to their processed image.
    """
    label_maps = []
    labels = []
    labels_out = {}

    # Iterate through all PNG images in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):  # Ensure only processing PNG files
            file_path = os.path.join(input_folder, file_name)

            # Load the image in grayscale
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Resize the image to the desired size
            img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)

            # Append the processed image to the label_maps list
            label_maps.append(img_resized)
            
            # Remove the .png extension from the file name and store it as a label (int)
            file_name_no_ext = os.path.splitext(file_name)[0]
            labels.append(file_name_no_ext)  # Store the image name (without extension)
            
            # Add the processed image to the dictionary with the label as the key
            labels_out[file_name_no_ext] = img_resized  # Use the name as key

    # Convert labels to a NumPy array (if necessary) 
    labels = np.array(labels)

    return label_maps, labels, labels_out

def process_images_with_opencv(input_folder, output_size=(256, 256)):
    """
    Process PNG images to label map format using OpenCV (grayscale, resized, and as NumPy arrays),
    and return corresponding labels (as integers) and labels_out.

    Parameters:
        input_folder (str): Path to the folder containing PNG images.
        output_size (tuple): Desired size for the output images (default: 256x256).

    Returns:
        list: A list of NumPy arrays representing processed label maps.
        np.array: Labels (integers) corresponding to the processed label maps.
        dict: labels_out mapping image names (without extensions) to their processed image.
    """
    label_maps = []

    # Iterate through all PNG images in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):  # Ensure only processing PNG files
            file_path = os.path.join(input_folder, file_name)

            # Load the image in grayscale
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Resize the image to the desired size
            img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)

            # Append the processed image to the label_maps list
            label_maps.append(img_resized)

    return label_maps



def load_dict_with_numpy(file_path):
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data.files}

def load_mask_dict(folder_path, file_prefix):
    # 构造完整文件路径
    file_path = os.path.join(folder_path, f"{file_prefix}.npz")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在！")
    
    # 加载 .npz 文件并转为字典
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data.files}





