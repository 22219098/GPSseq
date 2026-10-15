import tensorflow as tf
from tensorflow.keras.optimizers import AdamW, Adam       
from tensorflow.keras import layers, models
import pandas as pd
from tensorflow.keras.callbacks import Callback

import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray
from glob import glob
import os
import warnings
import imghdr
import struct
import h5py
from skimage.transform import resize
import os
from PIL import Image
from matplotlib.cm import get_cmap


def initialise_network(weights) -> Sequential:
    """
    Initialise a neural network with the given weights

    :param weights: The weights for the network
    :type weights: list
    :param species: The species of the animal, this is necessary because of a previous error where the models are slightly different for different species
    :return: The initialised neural network
    :rtype: keras.models.Sequential
    """
    weights = weights
    base_model = tf.keras.applications.ConvNeXtBase(include_top=False, pooling="avg")
    
    model = Sequential()
    model.add(base_model)
    model.add(Dense(1000, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dense(9, activation="linear",kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.load_weights(weights)
    return model


def gray_scale(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale

    :param img: The image to convert
    :type img: numpy.ndarray
    :return: The converted image
    :rtype: numpy.ndarray
    """
    img = rgb2gray(img).reshape(299, 299, 1)
    return img

def create_image_generator_with_customization(
    image_df,
    target_size=(224, 224),
    batch_size=1,
    color_mode="rgb",
    shuffle=False,
    samplewise_std_normalization=True,
    preprocessing_function=None,
):

    # Step 1: Create ImageDataGenerator and flow from DataFrame
    image_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        samplewise_std_normalization=samplewise_std_normalization,
    ).flow_from_dataframe(
        image_df,
        x_col="Filenames",
        y_col=None,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        shuffle=shuffle,
        class_mode=None,
    )
    filenames = image_generator.filenames  # 这是原始的 ImageDataGenerator 对象
    # Step 2: Wrap the generator using a custom generator function
    def custom_generator(generator):
        """
        Wrap the generator to ensure it outputs a tuple (input,) with tf.Tensor.
        """
        for batch in generator:
            # Convert batch to Tensor and wrap in a tuple
            yield (tf.convert_to_tensor(batch, dtype=tf.float32),)

    # Step 3: Convert the wrapped generator into a tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: custom_generator(image_generator),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, *target_size, 3), dtype=tf.float32),
        ),
    ).repeat()  # Set to repeat for consistent data feeding

    return dataset, filenames


def gray_scale_array(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale.

    :param img: The image to convert.
    :type img: numpy.ndarray
    :return: The converted image.
    :rtype: numpy.ndarray
    """
    img = rgb2gray(img).reshape(img.shape[0], img.shape[1], 1)  # 保持通用性
    return img


def create_image_array_generator_with_customization(
    image_array,  # 修复：确保参数名与调用时一致
    target_size=(224, 224),
    batch_size=1,
    color_mode="rgb",
    shuffle=False,
    samplewise_std_normalization=True,
    preprocessing_function=None,
):
    """
    Create a custom TensorFlow data generator pipeline from a numpy array.

    :param image_array: Numpy array representing image(s). Shape: (H, W, C) or (N, H, W, C).
    :param target_size: Tuple specifying the target size for images (width, height).
    :param batch_size: Batch size for the generator.
    :param color_mode: Color mode for the generator ('rgb' or 'grayscale').
    :param shuffle: Whether to shuffle the data.
    :param samplewise_std_normalization: Whether to apply sample-wise normalization.
    :param preprocessing_function: Function for image preprocessing.
    :return: A tf.data.Dataset object.
    """
    # Ensure the input is a numpy array
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a numpy array.")

    # If single image, add batch dimension
    if len(image_array.shape) == 3:  # Single image (H, W, C)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, H, W, C)

    # Handle grayscale conversion if required
    if color_mode == "grayscale" and image_array.shape[-1] == 3:  # If RGB but expecting grayscale
        image_array = np.mean(image_array, axis=-1, keepdims=True)

    # Resize all images to the target size
    resized_images = np.array([
        tf.image.resize(image, target_size).numpy() for image in image_array
    ])

    # Create ImageDataGenerator for numpy input
    image_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        samplewise_std_normalization=samplewise_std_normalization,
    ).flow(
        resized_images,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Wrap the generator to output tensors
    def custom_generator(generator):
        """
        Wrap the generator to ensure it outputs a tuple (input,) with tf.Tensor.
        """
        for batch in generator:
            # Convert batch to Tensor and wrap in a tuple
            yield (tf.convert_to_tensor(batch, dtype=tf.float32),)

    # Convert to tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: custom_generator(image_generator),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, *target_size, 1 if color_mode == "grayscale" else 3), dtype=tf.float32),
        ),
    ).repeat()  # Set to repeat for consistent data feeding

    # Return dataset and placeholder filenames
    filenames = ["input_array"] * len(resized_images)  # Placeholder filenames for all inputs
    return dataset, filenames


def predictions_util(
    model: Sequential,
    dataset,  # 直接传入 tf.data.Dataset
    weights: str,
    steps = 1,
    #species: str = "mouse",
    filenames=None
):
    model.load_weights(weights)
    predictions = model.predict(
        dataset,
        steps= steps,
        verbose=1,
    )
    predictions = predictions.astype(np.float64)

    filenames = [os.path.basename(i) for i in filenames]
    predictions_df = pd.DataFrame(
        {
            "Filenames": filenames,
            "ox": predictions[:, 0],
            "oy": predictions[:, 1],
            "oz": predictions[:, 2],
            "ux": predictions[:, 3],
            "uy": predictions[:, 4],
            "uz": predictions[:, 5],
            "vx": predictions[:, 6],
            "vy": predictions[:, 7],
            "vz": predictions[:, 8],
        }
    )
    return predictions_df

def generate_target_slice(alignment, volume):
    Ox, Oy, Oz, Ux, Uy, Uz, Vx, Vy, Vz = alignment
    
    Ox = Ox * 2.5
    Oy = 1320 - ((528 - Oy) * 2.5)
    Oz = 800 - ((320 - Oz) * 2.5)
    Ux = Ux * 2.5
    Uy = Uy * 2.5
    Uz = Uz * 2.5
    Vx = Vx * 2.5
    Vy = Vy * 2.5
    Vz = Vz * 2.5
    
    bounds = [1140, 1320, 800]

    X_size = int(round(np.sqrt(Ux**2 + Uy**2 + Uz**2)))
    Z_size = int(round(np.sqrt(Vx**2 + Vy**2 + Vz**2)))

    Uarange = np.linspace(0, 1, X_size, endpoint=False)
    Varange = np.linspace(0, 1, Z_size, endpoint=False)
    Ugrid, Vgrid = np.meshgrid(Uarange, Varange)

    Ugrid_x = Ugrid * Ux
    Ugrid_y = Ugrid * Uy
    Ugrid_z = Ugrid * Uz
    Vgrid_x = Vgrid * Vx
    Vgrid_y = Vgrid * Vy
    Vgrid_z = Vgrid * Vz

    X_Coords = (Ugrid_x + Vgrid_x).flatten() + Ox
    Y_Coords = (Ugrid_y + Vgrid_y).flatten() + Oy
    Z_Coords = (Ugrid_z + Vgrid_z).flatten() + Oz

    X_Coords = np.round(X_Coords).astype(int)
    Y_Coords = np.round(Y_Coords).astype(int)
    Z_Coords = np.round(Z_Coords).astype(int)

    out_of_bounds = (
        (X_Coords < 0) | (X_Coords >= bounds[0]) |
        (Y_Coords < 0) | (Y_Coords >= bounds[1]) |
        (Z_Coords < 0) | (Z_Coords >= bounds[2])
    )

    X_Coords[out_of_bounds] = 0
    Y_Coords[out_of_bounds] = 0
    Z_Coords[out_of_bounds] = 0

    regions = volume[X_Coords, Y_Coords, Z_Coords]
    regions = regions.reshape((Z_size, X_size))

    return regions



def generate_target_slice_voxel25(alignment, volume):
    Ox, Oy, Oz, Ux, Uy, Uz, Vx, Vy, Vz = alignment
    
    bounds = [456, 528, 320]

    X_size = int(round(np.sqrt(Ux**2 + Uy**2 + Uz**2)))
    Z_size = int(round(np.sqrt(Vx**2 + Vy**2 + Vz**2)))

    Uarange = np.linspace(0, 1, X_size, endpoint=False)
    Varange = np.linspace(0, 1, Z_size, endpoint=False)
    Ugrid, Vgrid = np.meshgrid(Uarange, Varange)

    Ugrid_x = Ugrid * Ux
    Ugrid_y = Ugrid * Uy
    Ugrid_z = Ugrid * Uz
    Vgrid_x = Vgrid * Vx
    Vgrid_y = Vgrid * Vy
    Vgrid_z = Vgrid * Vz

    X_Coords = (Ugrid_x + Vgrid_x).flatten() + Ox
    Y_Coords = (Ugrid_y + Vgrid_y).flatten() + Oy
    Z_Coords = (Ugrid_z + Vgrid_z).flatten() + Oz

    X_Coords = np.round(X_Coords).astype(int)
    Y_Coords = np.round(Y_Coords).astype(int)
    Z_Coords = np.round(Z_Coords).astype(int)

    out_of_bounds = (
        (X_Coords < 0) | (X_Coords >= bounds[0]) |
        (Y_Coords < 0) | (Y_Coords >= bounds[1]) |
        (Z_Coords < 0) | (Z_Coords >= bounds[2])
    )

    X_Coords[out_of_bounds] = 0
    Y_Coords[out_of_bounds] = 0
    Z_Coords[out_of_bounds] = 0

    regions = volume[X_Coords, Y_Coords, Z_Coords]
    regions = regions.reshape((Z_size, X_size))

    return regions





def convert_to_png_like_array(regions):
    """
    Simulates the effect of matplotlib saving a PNG and OpenCV reading it.

    Args:
        regions (np.ndarray): Input grayscale array (n, n).

    Returns:
        np.ndarray: RGB array (n, n, 3) simulating the PNG save and read process.
    """
    # Step 1: Normalize the data to the range [0, 255]
    # Simulate matplotlib's normalization behavior
    regions_min = regions.min()
    regions_max = regions.max()
    normalized = (regions - regions_min) / (regions_max - regions_min) * 255

    # Step 2: Convert to uint8
    regions_uint8 = normalized.astype(np.uint8)

    # Step 3: Expand to RGB (3 channels)
    regions_rgb = np.stack([regions_uint8, regions_uint8, regions_uint8], axis=2)

    return regions_rgb

def convert_to_png_like_array_with_tab20(regions):
    """
    Simulates the effect of matplotlib saving a PNG with cmap='tab20', vmin=0, vmax=2000.

    Args:
        regions (np.ndarray): Input grayscale array (n, n).

    Returns:
        np.ndarray: RGB array (n, n, 3) simulating the PNG save and read process.
    """
    # Step 1: Normalize the regions to the range [0, 1] based on vmin and vmax
    vmin, vmax = 0, 2000
    regions_clipped = np.clip(regions, vmin, vmax)  # Clip values outside [vmin, vmax]
    normalized = (regions_clipped - vmin) / (vmax - vmin)  # Normalize to [0, 1]

    # Step 2: Apply the tab20 colormap
    cmap = get_cmap('tab20')  # Get the tab20 colormap
    regions_rgb = cmap(normalized)  # Returns an (n, n, 4) array (RGBA)

    # Step 3: Convert RGBA to RGB by dropping the alpha channel
    regions_rgb = (regions_rgb[:, :, :3] * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    return regions_rgb