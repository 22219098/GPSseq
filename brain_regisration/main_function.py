import os
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import nrrd
import glob
import gc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import voxelmorph as vxm
from keras.utils import array_to_img
from .data.preprocess import preprocess_raw_image, scale, scale_from_array, check_image_group, parse_brain_regions
from .models.genG import get_resnet_generator
from .models.affine_generator import image_affine_with_ConvNext, Affine_label, process_with_affine
from .utils.visualize import overlay_images_with_transparency, overlay_images, brain_region_contour_overlay, total_brain_contour_overlay, hex_to_bgr, colorize
from .models.ConvNext import gray_scale_array, create_image_array_generator_with_customization, predictions_util, generate_target_slice, convert_to_png_like_array, convert_to_png_like_array_with_tab20, generate_target_slice_voxel25
from .angle_integration.angle_methods import propagate_angles
from .initialize_models import initialize_models
from scipy.ndimage import binary_dilation
from .utils.hemisphere import process_brain_image_pipeline, crop_half_by_centroid_and_axis, mask_half_image

class BrainRegistration:
    def __init__(self, voxel: str = 'nii'):
        self.voxel = voxel
        # 调用 initialize_models 并存储结果
        (self.df,
         self.volume_nii, 
         self.annotation_nii,
         self.annotation_boundary_nii,
         self.channels,
         self.gen_G_bf,
         self.gen_G_df,
         self.ConvNext, 
         self.ConvNext_weights, 
         self.mask_nii_folder, 
         self.model_aff, 
         self.vxm_model) = initialize_models(voxel=voxel)


    def mouse_brain_registration(
        self,
        raw_image_path,
        output_dir,
        section_layer,
        brain_region,
        group = False,
        save_temp=False,
        temp_result = True,
        display_mode=("block", "#399335"),
        enlarge = False,
        brain_state = "complete",  # "damaged", "hemisphere"
        angle_integration=False
    ):
        """Complete pipeline: Style transfer, affine registration, U-Net registration, and visualization."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        mode, hex_color = display_mode
        bgr_color = hex_to_bgr(hex_color)

        #check address folder or images
        group = check_image_group(raw_image_path)
        
        #prepare for brain region
        region_mask_data = []
        brain_regions = []   
        
        ##############Group##############
        if angle_integration and group is None:
            raise ValueError("angle_integration can only be enabled when group is True.")
        if group:
            #Ensure auto registration is on
            if angle_integration and section_layer is not None:
                raise ValueError("angle_integration can only be enabled when section_layer is None.")
            if angle_integration and brain_region is not None:
                raise ValueError("angle_integration can only be enabled when brain_region is None.")
    
            ##################angle_integration#############
            ####### Step 1: Initialize step1_predictions container  ############
            #if not hasattr(self, 'step1_predictions'):  # Check if the container exists
            self.step1_predictions = []
        
            # Get the list of images in the directory
            jpg_files = glob.glob(os.path.join(raw_image_path, "*.jpg"))
            png_files = glob.glob(os.path.join(raw_image_path, "*.png"))
            image_files = jpg_files + png_files  # 合并两种格式的文件
            image_output_dirs = []
            # Process images
            for img_path in image_files:
                # Step 1: Style Transfer
                img_tensor, field = preprocess_raw_image(img_path)  # Pass the correct image path
                img_tensor = tf.expand_dims(img_tensor, axis=0)
                if field == "dark_field":
                    prediction = self.gen_G_df(img_tensor, training=False)[0].numpy()
                elif field == "bright_field":
                    prediction = self.gen_G_bf(img_tensor, training=False)[0].numpy()
                prediction = (prediction + 1.0) / 2.0
                prediction = np.clip(prediction, 0, 1)  # 确保在 [0,1] 之间
                step1_prediction = (prediction * 255).astype(np.uint8)
                step1_prediction = np.repeat(step1_prediction, 3, axis=-1)
                
                # 创建一个新的文件夹，基于当前索引或其他唯一标识符
                original_filename = os.path.basename(img_path)
                filename_without_extension = os.path.splitext(original_filename)[0]
                image_output_dir = os.path.join(output_dir, f"{filename_without_extension}")  # 为每张图像创建唯一的文件夹
                os.makedirs(image_output_dir, exist_ok=True)  # 如果文件夹已存在，跳过创建
                image_output_dirs.append(image_output_dir)

                # Append prediction to the list
                self.step1_predictions.append(step1_prediction)
                # If we have enough predictions, stop the loop
                if len(self.step1_predictions) >= len(image_files):
                    break
                
            # Check if angle_integration and brain_state
            if angle_integration and brain_state == "hemisphere":
                raise ValueError(f"Hemisphere model can not use angle_integration")

            ###########step1.5 Layer Registration  ##############
            elif brain_state == "hemisphere" and not angle_integration:
                processed_predictions = []
                coords_list = []
                center_list = []
                hemisphere_list = []

                for step1_prediction in self.step1_predictions:

                    # 提取单通道
                    step1_prediction = step1_prediction[:, :, 0]

                    # 调用处理管道
                    step1_prediction, (px1, py1, px2, py2), (cx, cy), hemisphere = process_brain_image_pipeline(step1_prediction)

                    # 恢复为三通道
                    step1_prediction = np.repeat(step1_prediction[..., np.newaxis], 3, axis=-1)

                    # 保存结果
                    processed_predictions.append(step1_prediction)
                    coords_list.append((px1, py1, px2, py2))
                    center_list.append((cx, cy))
                    hemisphere_list.append(hemisphere)

                if section_layer is None: 
                    all_predictions = []
                    for idx, step1_pred in enumerate(processed_predictions):
                    
                        dataset, filenames = create_image_array_generator_with_customization(
                            image_array= step1_pred,
                            target_size=(224, 224),
                            batch_size=1,
                            color_mode="rgb",  # 可选 'rgb' 或 'grayscale'
                            preprocessing_function=gray_scale_array,
                        )
                    
                        predictions_df = predictions_util(
                            model=self.ConvNext,
                            dataset=dataset,
                            weights=self.ConvNext_weights,
                            steps = 1 // 1, 
                            filenames=filenames
                        )
                        all_predictions.append(predictions_df)
                    result = pd.concat(all_predictions, ignore_index=True)

                elif isinstance(section_layer, str):
                
                    filtered_df = self.df[self.df['Layer'].astype(str) == section_layer]
                    subset_df = filtered_df.iloc[:, 0:7]
                    result = pd.concat([subset_df] * len(image_files), ignore_index=True)

                else:
                    raise ValueError("Invalid section_layer! Must be an str or None.")    


                for idx, row in result.iterrows():
                    
                    if section_layer is None:
                        # 提取每一行的 alignment 信息
                        alignment = row[1:].values.tolist()
                        # 提取对应的对称轴和质点信息
                        px1, py1, px2, py2 = coords_list[idx]
                        cx, cy = center_list[idx] 

                        alignment = predictions_df.iloc[0, 1:].values.tolist()
                        alignment_df = pd.DataFrame({
                            'alignment': [alignment],
                            'px1': [px1],
                            'py1': [py1],
                            'px2': [px2],
                            'py2': [py2],
                            'cx': [cx],
                            'cy': [cy],
                            'hemisphere': [hemisphere]
                        })
                        alignment_df.to_csv(os.path.join(image_output_dirs[idx], "alignment_df.csv"))

                        # 生成对应的 ConvNext_template 和 ConvNext_annotation
                        if self.voxel == "nii":
                            ConvNext_template = generate_target_slice(alignment, self.volume_nii)
                        elif self.voxel == "nrrd":
                            ConvNext_template = generate_target_slice_voxel25(alignment, self.volume_nii)
                        ConvNext_template = convert_to_png_like_array(ConvNext_template)
                        ConvNext_template = cv2.resize(ConvNext_template, (256, 256), interpolation=cv2.INTER_AREA)
                        # 从保存的列表中取出与当前 index 匹配的裁剪坐标
                        ConvNext_template = crop_half_by_centroid_and_axis(ConvNext_template, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0))  # 黑色填充

                        if self.voxel == "nii":
                            ConvNext_annotation = generate_target_slice(alignment, self.annotation_nii)
                        elif self.voxel == "nrrd":
                            ConvNext_annotation = generate_target_slice_voxel25(alignment, self.annotation_nii)
                        if mode == "block" or mode == "line":
                            ConvNext_annotation = convert_to_png_like_array_with_tab20(ConvNext_annotation)
                            annotation_resized = cv2.resize(ConvNext_annotation, (256, 256), interpolation=cv2.INTER_AREA)
                            annotation_resized = crop_half_by_centroid_and_axis(annotation_resized, px1, py1, px2, py2, cx, cy, fill_value=(17, 110, 175))
                        if mode == "Allen":
                            if self.voxel == "nii":
                                ConvNext_annotation = colorize(self.channels, ConvNext_annotation, 'substructure')
                            elif self.voxel == "nrrd":
                                ConvNext_annotation = colorize(self.channels, ConvNext_annotation, 'ARA structure ID')
                            annotation_resized = cv2.resize(ConvNext_annotation, (256, 256), interpolation=cv2.INTER_AREA)
                            annotation_resized = crop_half_by_centroid_and_axis(annotation_resized, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0))  # 蓝色填充
                        
                        ConvNext_boundary = generate_target_slice(alignment, self.annotation_boundary_nii)
                        ConvNext_boundary = convert_to_png_like_array(ConvNext_boundary)
                        boundary_resized = cv2.resize(ConvNext_boundary, (256, 256), interpolation=cv2.INTER_AREA)
                        boundary_resized = crop_half_by_centroid_and_axis(boundary_resized, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0))  # 黑色填充

                        if isinstance(brain_region, str) and brain_region.strip():
                            brain_regions = parse_brain_regions(brain_region)
                            for region in brain_regions:
                                if self.voxel == "nii":
                                    mask_nii_image = sitk.ReadImage(os.path.join(self.mask_nii_folder, f"{region}.nii.gz"))
                                    mask_nii = sitk.GetArrayViewFromImage(mask_nii_image)
                                    mask_nii = np.transpose(mask_nii, (0, 2, 1))[::-1, ::-1, ::-1]
                                    ConvNext_mask = generate_target_slice(alignment, mask_nii)

                                elif self.voxel == "nrrd":
                                    mask_nii, _ = nrrd.read(os.path.join(self.mask_nii_folder, f"{region}.nrrd"))
                                    mask_nii = np.transpose(mask_nii, (2, 0, 1))[:, ::-1, ::-1]
                                    ConvNext_mask = generate_target_slice_voxel25(alignment, mask_nii)

                                ConvNext_mask_npy = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                ConvNext_mask = convert_to_png_like_array(ConvNext_mask)
                                mask_resized = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                mask_resized = crop_half_by_centroid_and_axis(
                                    mask_resized, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0)
                                )
                                region_mask_data.append({
                                    "region": region,
                                    "mask_resized": mask_resized,
                                    "ConvNext_mask_npy": ConvNext_mask_npy
                                })
                        else:
                            region_mask_data = []



                    if section_layer is not None:
                        # 提取每一行的 alignment 信息
                        alignment = row[2]
                        px1, py1, px2, py2 = coords_list[idx]
                        cx, cy = center_list[idx]
                        hemisphere = hemisphere_list[idx]

                        if brain_state == "hemisphere":
                            alignment_df = pd.DataFrame({
                                'alignment': [alignment],
                                'px1': [px1],
                                'py1': [py1],
                                'px2': [px2],
                                'py2': [py2],
                                'cx': [cx],
                                'cy': [cy],
                                'hemisphere': [hemisphere]
                            })
                        alignment_df.to_csv(os.path.join(image_output_dirs[idx], "alignment_df.csv"))

    
                        # 生成对应的 ConvNext_template 和 ConvNext_annotation
                        slice_y = self.volume_nii[:, alignment, :]
                        rotated_slice = np.rot90(slice_y, k=2)  # 旋转180度
                        transposed_slice = rotated_slice.T  # 转置操作
                        ConvNext_template = convert_to_png_like_array(transposed_slice)
                        ConvNext_template = cv2.resize(ConvNext_template, (256, 256), interpolation=cv2.INTER_AREA)
                        annotation_y = self.annotation_nii[:, alignment, :]
                        rotated_annotation = np.rot90(annotation_y, k=2)  # 旋转180度
                        transposed_annotation = rotated_annotation.T  # 转置操作
                        if mode == "block":
                            ConvNext_annotation = convert_to_png_like_array_with_tab20(transposed_annotation)
                        if mode == "Allen":
                            if self.voxel == "nii":
                                ConvNext_annotation = colorize(self.channels, transposed_annotation, 'substructure')
                            elif self.voxel == "nrrd":
                                ConvNext_annotation = colorize(self.channels, transposed_annotation, 'ARA structure ID')
                        annotation_resized = cv2.resize(ConvNext_annotation, (256, 256), interpolation=cv2.INTER_AREA)

                        boundary_y = self.annotation_boundary_nii[:, alignment, :]
                        rotated_boundary = np.rot90(boundary_y, k=2)  # 旋转180度
                        transposed_boundary = rotated_boundary.T  # 转置操作
                        ConvNext_boundary = convert_to_png_like_array(transposed_boundary)     
                        boundary_resized = cv2.resize(ConvNext_boundary, (256, 256), interpolation=cv2.INTER_AREA)

                        if brain_state == "hemisphere":
                            ConvNext_template = mask_half_image(ConvNext_template, side= hemisphere, fill_color=(0, 0, 0))
                            if mode == "block":
                                annotation_resized = mask_half_image(annotation_resized, side= hemisphere, fill_color=(17, 110, 175))
                            if mode == "Allen":
                                annotation_resized = mask_half_image(annotation_resized, side= hemisphere, fill_color=(0, 0, 0))
                            boundary_resized = mask_half_image(boundary_resized, side= hemisphere, fill_color=(0, 0, 0))                
                        
                    
                        if isinstance(brain_region, str) and brain_region.strip():
                            brain_regions = parse_brain_regions(brain_region)
                            for region in brain_regions:
                                if self.voxel == "nii":
                                    mask_nii_image = sitk.ReadImage(os.path.join(self.mask_nii_folder, f"{region}.nii.gz"))
                                    mask_nii = sitk.GetArrayViewFromImage(mask_nii_image)
                                    mask_nii = np.transpose(mask_nii, (0, 2, 1))[::-1, ::-1, ::-1]
                                elif self.voxel == "nrrd":
                                    mask_nii, _ = nrrd.read(os.path.join(self.mask_nii_folder, f"{region}.nrrd"))
                                    mask_nii = np.transpose(mask_nii, (2, 0, 1))[:, ::-1, ::-1]
                                
                                mask_y = mask_nii[:, alignment, :]
                                rotated_mask = np.rot90(mask_y, k=2)  # 旋转180度
                                transposed_mask = rotated_mask.T  # 转置操作
                                ConvNext_mask_npy = cv2.resize(transposed_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                ConvNext_mask = convert_to_png_like_array(transposed_mask)
                                mask_resized = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                mask_resized = mask_half_image(mask_resized, side= hemisphere, fill_color=(0, 0, 0))
                                region_mask_data.append({
                                    "region": region,
                                    "mask_resized": mask_resized,
                                    "ConvNext_mask_npy": ConvNext_mask_npy
                                }) 
                        else:
                            region_mask_data = []
                    # 获取 RGB 图像
                    rgb_image = annotation_resized
                    
                    # 获取对应的 step1_prediction
                    step1_pred = self.step1_predictions[idx]  # 从 step1_predictions 中选择对应的预测图像
                            
                    # 调用 image_affine_with_ConvNext 来生成相应的结果
                    source_test, target_test, in_shape = image_affine_with_ConvNext(step1_pred, ConvNext_template)

                    # Step 2: Affine Registration    
                    source_ima, source_map = process_with_affine(source_test)
                    target_ima, target_map = process_with_affine(target_test)

                    trans_test = self.model_aff.predict((source_map, target_map))
                    affine_field = trans_test[0]

                    moved_ima = vxm.layers.SpatialTransformer(fill_value=0)((source_ima, trans_test))
                    template = tf.squeeze(moved_ima, axis=0)
                    step2_template = tf.cast(template, dtype=tf.uint8)

                    if save_temp:
                        #step1 result saved
                        step1_output_path = os.path.join(image_output_dirs[idx], f"step1_predicted.png")
                        array_to_img(self.step1_predictions[idx]).save(step1_output_path)  # 保存预测结果
                        #step2 result saved
                        np.save(os.path.join(image_output_dirs[idx], "step2_affine_field.npy"), affine_field)
                        encoded_template = tf.image.encode_png(step2_template)
                        tf.io.write_file(os.path.join(image_output_dirs[idx], "step2_template.png"), encoded_template)

                    # Step2.5 Block & Mask
                    if brain_state != "complete":
                        binary_step1_prediction = (self.step1_predictions[idx] > 15).astype(np.uint8)
                        # 对 binary1 进行膨胀操作（向上下左右扩展 5 像素）
                        gray_mask = binary_step1_prediction[:, :, 0]
                        structure = np.ones((11, 11))  # 5 pixels in all directions → kernel size = 2*5 + 1
                        expanded_mask = binary_dilation(gray_mask, structure=structure).astype(np.uint8)
                        expanded_mask_3d = expanded_mask[..., np.newaxis]
                        # 使用 mask 筛选 binary2 / 原图
                        step2_template = step2_template * expanded_mask_3d  # 如果你想应用到原图像素

                        
                    # Step 3: U-Net Registration
                    moving_image = scale_from_array(step2_template)
                    fixed_image = scale_from_array(self.step1_predictions[idx])

                    moving_image = np.array(moving_image) / 255.0
                    fixed_image = np.array(fixed_image) / 255.0

                    moving_image = moving_image[np.newaxis, ..., np.newaxis]
                    fixed_image = fixed_image[np.newaxis, ..., np.newaxis]

                    val_pred = self.vxm_model.predict((moving_image, fixed_image))
                    displacement_field = val_pred[1][0]

                    if save_temp:
                        np.save(os.path.join(image_output_dirs[idx], "step3_displacement_field.npy"), displacement_field)

                    # Step 4: Combine and Visualize
                    total_displacement_field = affine_field + displacement_field
                    raw_image = cv2.imread(image_files[idx])
                    original_shape = raw_image.shape[:-1]
                    target_shape = (original_shape[1], original_shape[0])

                    resized_fixed_image = cv2.resize(self.step1_predictions[idx], target_shape, interpolation=cv2.INTER_LINEAR)

                    # Moving template
                    new_template = scale_from_array(step2_template)
                    new_template = new_template.astype('float32') / 255.0
                    new_template = new_template[..., np.newaxis]

                    # Moving annotation
                    R_channel = rgb_image[:, :, 0].astype('float32') / 255.0
                    G_channel = rgb_image[:, :, 1].astype('float32') / 255.0
                    B_channel = rgb_image[:, :, 2].astype('float32') / 255.0
            
                    R_channel = R_channel[..., np.newaxis]
                    G_channel = G_channel[..., np.newaxis]
                    B_channel = B_channel[..., np.newaxis]

                    # Moving boundary
                    new_boundary_R = boundary_resized[:, :, 0].astype('float32') / 255.0
                    new_boundary_G = boundary_resized[:, :, 1].astype('float32') / 255.0
                    new_boundary_B = boundary_resized[:, :, 2].astype('float32') / 255.0

                    new_boundary_R = new_boundary_R[..., np.newaxis]
                    new_boundary_G = new_boundary_G[..., np.newaxis]
                    new_boundary_B = new_boundary_B[..., np.newaxis]
            
                    # SpatialTransformer
                    spatial_transformer = vxm.layers.SpatialTransformer()

                    # Template
                    transformed_template = spatial_transformer(
                        [new_template[np.newaxis, ...], displacement_field[np.newaxis, ...]]
                    )
                    transformed_template = transformed_template[0, ..., 0]

                    if brain_state != "complete":
                        mask_transformed_template = (transformed_template.numpy() > 4 / 255).astype(np.uint8)
                        mask_3d = mask_transformed_template[..., np.newaxis]  # shape [H, W, 1]

                    resized_transformed_template = cv2.resize(
                        transformed_template.numpy(), target_shape, interpolation=cv2.INTER_LINEAR
                    )


                    # Annotation
                    transformed_R_channel = spatial_transformer(
                        [R_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    )
                    transformed_G_channel = spatial_transformer(
                        [G_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
                    transformed_B_channel = spatial_transformer(
                        [B_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
                    transformed_R_channel = transformed_R_channel[0, ..., 0] * 255
                    transformed_G_channel = transformed_G_channel[0, ..., 0] * 255
                    transformed_B_channel = transformed_B_channel[0, ..., 0] * 255

                    # Boundary
                    transformed_boundary_R = spatial_transformer(
                        [new_boundary_R[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    )
                    transformed_boundary_G = spatial_transformer(
                        [new_boundary_G[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
                    transformed_boundary_B = spatial_transformer(
                        [new_boundary_B[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
            
                    transformed_boundary_R = transformed_boundary_R[0, ..., 0] * 255
                    transformed_boundary_G = transformed_boundary_G[0, ..., 0] * 255
                    transformed_boundary_B = transformed_boundary_B[0, ..., 0] * 255
                    # Save displacement_field
                    np.save(os.path.join(image_output_dirs[idx], "total_displacement_field.npy"), total_displacement_field)
            
            
            
                    # 保存 resized_transformed_template & resized_transformed_annotation
                    resized_transformed_template_uint8 = (resized_transformed_template * 255).astype(np.uint8)
                    rgb_moved = np.stack((transformed_R_channel, transformed_G_channel, transformed_B_channel), axis=2).astype(np.uint8)
                    if mode == "block":
                        target_color = np.array([17, 110, 175], dtype=np.uint8)
                        color_layer = np.ones_like(rgb_moved) * target_color
                        result_rgb = rgb_moved * mask_3d + color_layer * (1 - mask_3d)
                        rgb_moved = result_rgb.astype(np.uint8)
                    elif mode == "Allen":
                        target_color = np.array([0, 0, 0], dtype=np.uint8)
                        color_layer = np.ones_like(rgb_moved) * target_color
                        result_rgb = rgb_moved * mask_3d + color_layer * (1 - mask_3d)
                        rgb_moved = result_rgb.astype(np.uint8)
                    resized_transformed_annotation = cv2.resize(rgb_moved, target_shape, interpolation=cv2.INTER_LINEAR)
                    boundary_moved = np.stack((transformed_boundary_R, transformed_boundary_G, transformed_boundary_B), axis=2).astype(np.uint8)
                    target_color = np.array([0, 0, 0], dtype=np.uint8)
                    color_layer = np.ones_like(boundary_moved) * target_color  # shape [H, W, 3]
                    result_boundary = boundary_moved * mask_3d + color_layer * (1 - mask_3d)
                    boundary_moved = result_boundary.astype(np.uint8)
                    resized_transformed_boundary = cv2.resize(boundary_moved, target_shape, interpolation=cv2.INTER_LINEAR)
                    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                    raw_image_diaplay = raw_image_rgb
            
                    # enlarge or not
                    if enlarge:
                        binary_map = np.where(resized_transformed_template_uint8 > 15, 1.0, 0.0).astype(np.uint8)  # 二值化
                        rows, cols = np.where(binary_map == 1)
                        # 确定上下左右边界
                        top = np.min(rows)    # 最小行号 -> 上边界
                        bottom = np.max(rows) # 最大行号 -> 下边界
                        left = np.min(cols)   # 最小列号 -> 左边界
                        right = np.max(cols)  # 最大列号 -> 右边界
                        resized_transformed_template_uint8 = resized_transformed_template_uint8[top:bottom+1, left:right+1]
                        resized_transformed_annotation = resized_transformed_annotation[top:bottom+1, left:right+1]
                        resized_transformed_boundary = resized_transformed_boundary[top:bottom+1, left:right+1]
                        raw_image_rgb = raw_image_rgb[top:bottom+1, left:right+1]
                        raw_image = raw_image[top:bottom+1, left:right+1]

                    # save
                    plt.imsave(os.path.join(image_output_dirs[idx], "resized_transformed_template.png"), resized_transformed_template_uint8, cmap='gray')
                    plt.imsave(os.path.join(image_output_dirs[idx], "resized_transformed_annotation.png"), resized_transformed_annotation)
                    plt.imsave(os.path.join(image_output_dirs[idx], "resized_transformed_boundary.png"), resized_transformed_boundary, cmap='gray')
                    
                    
                    #visualize
                    if mode == "block":
                        overlay = cv2.resize(resized_transformed_annotation, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                        Total_overlay = overlay_images(raw_image_rgb, resized_transformed_annotation, alpha=0.3)
                        result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(image_output_dirs[idx], "Total_brain_registration.png"), result_bgr)

                    if mode == "Allen":
                        overlay = cv2.resize(resized_transformed_annotation, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                        Total_overlay = overlay_images(raw_image_rgb, resized_transformed_annotation, alpha=0.3)
                        result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(image_output_dirs[idx], "Total_brain_registration.png"), result_bgr)

                    if mode == "line":
                        overlay = cv2.resize(resized_transformed_boundary, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                        Total_overlay = overlay_images(raw_image_rgb, resized_transformed_boundary, alpha=0.3)
                        result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(image_output_dirs[idx], "Total_brain_registration.png"), result_bgr)
                        
                    file_name = os.path.basename(image_output_dirs[idx])
                    
                    if  temp_result:         
                        plt.figure(figsize=(18, 5))

                        # Raw Image
                        plt.subplot(1, 4, 1)
                        plt.imshow(raw_image_diaplay)
                        plt.title("Raw Image")
                        plt.axis("off")

                        # CycleGAN Predicted
                        plt.subplot(1, 4, 2)
                        plt.imshow(resized_fixed_image, cmap="gray")
                        plt.title("CycleGAN Predicted")
                        plt.axis("off")

                        # Template Image
                        plt.subplot(1, 4, 3)
                        plt.imshow(resized_transformed_template, cmap="gray")
                        plt.title("Brain Template")
                        plt.axis("off")

                        # Annotation
                        plt.subplot(1, 4, 4)
                        plt.imshow(resized_transformed_annotation)
                        plt.title("Annotation")
                        plt.axis("off")
                        
                        plt.tight_layout()
                        plt.close() 
                        
                        plt.figure(figsize=(12,8))          
                
                        plt.subplot(1, 1, 1)
                        plt.imshow(Total_overlay)
                        plt.title(f"{file_name} registration", fontsize=20)
                        plt.axis("off")    
                        
                        plt.tight_layout()
                        plt.close()     
                        
                    else:
                
                        plt.figure(figsize=(12,8))          
                
                        plt.subplot(1, 1, 1)
                        plt.imshow(Total_overlay)
                        plt.title(f"{file_name} registration", fontsize=20)
                        plt.axis("off") 
                        
                        plt.tight_layout()
                        plt.close()     
                    
                    
                    
                    if region_mask_data:
                        for item in region_mask_data:
                            region = item["region"]
                            mask_resized = item["mask_resized"]
                            ConvNext_mask_npy = item["ConvNext_mask_npy"]
                            
                            #moving brain region
                            mask = scale_from_array(mask_resized)  # 灰度图
                            mask = mask.astype('float32') / 255.0  # 归一化到 [0, 1]
                            mask = mask[..., np.newaxis]  # 添加通道维度，shape: [H, W, 1]
                            mask_npy = ConvNext_mask_npy[..., np.newaxis]

                            transformed_mask = spatial_transformer([mask[np.newaxis, ...], total_displacement_field[np.newaxis, ...]])
                            transformed_mask = transformed_mask[0, ..., 0]  # shape: [H, W]
                            transformed_mask_np = transformed_mask.numpy()

                            mask_npy = mask_npy.astype(np.float32)
                            transformed_mask_npy = spatial_transformer([mask_npy[np.newaxis, ...], total_displacement_field[np.newaxis, ...]])
                            transformed_mask_npy = transformed_mask_npy[0, ..., 0]  # shape: [H, W]
                            transformed_mask_npy_np = transformed_mask_npy.numpy()


                            resized_transformed_mask = cv2.resize(transformed_mask_np, target_shape, interpolation=cv2.INTER_LINEAR)
                            resized_transformed_mask_uint8 = (resized_transformed_mask * 255).astype(np.uint8)
                            resize_mask_npy = cv2.resize(transformed_mask_npy_np, target_shape, interpolation=cv2.INTER_LINEAR)
                            resize_mask_npy = (resize_mask_npy > 0.5).astype(np.uint8)
                    
                            if enlarge:
                                resized_transformed_mask_uint8 = resized_transformed_mask_uint8[top:bottom+1, left:right+1]
                                resize_mask_npy = resize_mask_npy[top:bottom+1, left:right+1]
                    
                            # 保存 resized_transformed_mask
                            plt.imsave(os.path.join(image_output_dirs[idx], f"{region}_resized_transformed_mask.png"), resized_transformed_mask_uint8, cmap='gray')  # 灰度图像
                            np.save(os.path.join(image_output_dirs[idx], f"{region}_resized_transformed_mask.npy"), resize_mask_npy)
                
                            # 调用函数，使用黑色阈值去掉接近黑色的边缘
                            if mode == "block" or mode == "Allen":
                                resized_raw_image_uint8 = raw_image.astype(np.uint8)
                                combined_image = overlay_images_with_transparency(
                                background_array=resized_raw_image_uint8,  # 背景图 (H, W, 3)
                                overlay_array=resized_transformed_mask_uint8,  # 叠加图 (H, W)
                                alpha=0.5,  # 透明度
                                black_threshold=100,  # 设置黑色阈值（去掉小于 20 的灰度值）
                                output_path=os.path.join(image_output_dirs[idx], f"{region}_registration.png")
                                )
                            if mode == "line":
                                combined_image = brain_region_contour_overlay(resized_transformed_mask_uint8, raw_image, bgr_color)
                                cv2.imwrite(os.path.join(image_output_dirs[idx], f"{region}_registration.png"), combined_image)
                                combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        

                            if temp_result:
                                # Brain Region (mask_resized)
                                plt.figure(figsize=(12, 8))
                                
                                plt.subplot(1, 2, 1)
                                plt.imshow(resized_transformed_mask, cmap="gray")
                                plt.title(f"Brain Region: {region}")
                                plt.axis("off")
                        
                                plt.subplot(1, 2, 2)
                                plt.imshow(combined_image)
                                plt.title(f"Brain region registration: {region}", fontsize=20)
                                plt.axis("off")
                        
                                plt.tight_layout()
                                plt.close()  

                            else:
                        
                                plt.figure(figsize=(12,8))             
                                plt.imshow(combined_image)
                                plt.title(f"Brain region registration: {region}", fontsize=20)
                                plt.axis("off")
                        
                                plt.tight_layout()
                                plt.close()  
            
                    gc.collect()                
                    


            # 如果 step1_predictions 数量足够（大于或等于10个），继续处理
            # 执行后续步骤：例如 layer registration、U-Net registration 等
            else:

                if section_layer is None: 
                    all_predictions = []
                    for idx, step1_pred in enumerate(self.step1_predictions):
                    
                        dataset, filenames = create_image_array_generator_with_customization(
                            image_array= step1_pred,
                            target_size=(224, 224),
                            batch_size=1,
                            color_mode="rgb",  # 可选 'rgb' 或 'grayscale'
                            preprocessing_function=gray_scale_array,
                        )
                    
                        predictions_df = predictions_util(
                            model=self.ConvNext,
                            dataset=dataset,
                            weights=self.ConvNext_weights,
                            steps = 1 // 1, 
                            filenames=filenames
                        )
                        all_predictions.append(predictions_df)
                    result = pd.concat(all_predictions, ignore_index=True)
                
                
                elif isinstance(section_layer, str):
                    
                    filtered_df = self.df[self.df['Layer'].astype(str) == section_layer]
                    subset_df = filtered_df.iloc[:, 0:7]
                    result = pd.concat([subset_df] * len(image_files), ignore_index=True)
                
                
                
                else:
                    raise ValueError("Invalid section_layer! Must be an str or None.")

                
                # angle_integration
                # Check if we have at least 10 prediction
                if angle_integration and len(self.step1_predictions) < 10:
                    raise ValueError(f"Not enough images processed. Only {len(self.step1_predictions)} processed. At least 10 are required.")
                elif angle_integration and len(self.step1_predictions) >= 10:
                    result = propagate_angles(result, "weighted_mean", "mouse")
                                
                # Generate ConvNext_template, annotation_resized
                for idx, row in result.iterrows():
                    
                    if section_layer is None:
                        # 提取每一行的 alignment 信息
                        alignment = row[1:].values.tolist()

                        alignment_df = pd.DataFrame({'alignment': [alignment]})
                        alignment_df.to_csv(os.path.join(image_output_dirs[idx], "alignment_df.csv"))
                    
                        # 生成对应的 ConvNext_template 和 ConvNext_annotation
                        if self.voxel == "nii":
                            ConvNext_template = generate_target_slice(alignment, self.volume_nii)
                            ConvNext_annotation = generate_target_slice(alignment, self.annotation_nii)
                        elif self.voxel == "nrrd":
                            ConvNext_template = generate_target_slice_voxel25(alignment, self.volume_nii)
                            ConvNext_annotation = generate_target_slice_voxel25(alignment, self.annotation_nii)

                        ConvNext_template = convert_to_png_like_array(ConvNext_template)
                        ConvNext_template = cv2.resize(ConvNext_template, (256, 256), interpolation=cv2.INTER_AREA)
                        
                        if mode == "block" or mode == "line":
                            ConvNext_annotation = convert_to_png_like_array_with_tab20(ConvNext_annotation)
                        if mode == "Allen":
                            if self.voxel == "nii":
                                ConvNext_annotation = colorize(self.channels, ConvNext_annotation, 'substructure')
                            elif self.voxel == "nrrd":
                                ConvNext_annotation = colorize(self.channels, ConvNext_annotation, 'ARA structure ID')
                        ConvNext_boundary = generate_target_slice(alignment, self.annotation_boundary_nii)
                        ConvNext_boundary = convert_to_png_like_array(ConvNext_boundary)
                        
                        if isinstance(brain_region, str) and brain_region.strip():
                            brain_regions = parse_brain_regions(brain_region)
                            for region in brain_regions:
                                if self.voxel == "nii":
                                    mask_nii_image = sitk.ReadImage(os.path.join(self.mask_nii_folder, f"{region}.nii.gz"))
                                    mask_nii = sitk.GetArrayViewFromImage(mask_nii_image)
                                    mask_nii = np.transpose(mask_nii, (0, 2, 1))[::-1, ::-1, ::-1]
                                    ConvNext_mask = generate_target_slice(alignment, mask_nii)
                                elif self.voxel == "nrrd":
                                    mask_nii, _ = nrrd.read(os.path.join(self.mask_nii_folder, f"{region}.nrrd"))
                                    mask_nii = np.transpose(mask_nii, (2, 0, 1))[:, ::-1, ::-1]
                                    ConvNext_mask = generate_target_slice_voxel25(alignment, mask_nii)

                                ConvNext_mask_npy = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                ConvNext_mask = convert_to_png_like_array(ConvNext_mask)
                                mask_resized = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                region_mask_data.append({
                                    "region": region,
                                    "mask_resized": mask_resized,
                                    "ConvNext_mask_npy": ConvNext_mask_npy
                                })
                        else:
                            region_mask_data = []
  
                        
                                        
                        
                    if section_layer is not None:
                        # 提取每一行的 alignment 信息
                        alignment = row[2]

                        alignment_df = pd.DataFrame({'alignment': [alignment]})
                        alignment_df.to_csv(os.path.join(image_output_dirs[idx], "alignment_df.csv"))
    
                        # 生成对应的 ConvNext_template 和 ConvNext_annotation
                        slice_y = self.volume_nii[:, alignment, :]
                        rotated_slice = np.rot90(slice_y, k=2)  # 旋转180度
                        transposed_slice = rotated_slice.T  # 转置操作
                        ConvNext_template = convert_to_png_like_array(transposed_slice)
                        ConvNext_template = cv2.resize(ConvNext_template, (256, 256), interpolation=cv2.INTER_AREA)

                        annotation_y = self.annotation_nii[:, alignment, :]
                        rotated_annotation = np.rot90(annotation_y, k=2)  # 旋转180度
                        transposed_annotation = rotated_annotation.T  # 转置操作
                        if mode == "block":
                            ConvNext_annotation = convert_to_png_like_array_with_tab20(transposed_annotation)
                        if mode == "Allen":
                            if self.voxel == "nii":
                                ConvNext_annotation = colorize(self.channels, transposed_annotation, 'substructure')
                            if self.voxel == "nrrd":
                                ConvNext_annotation = colorize(self.channels, transposed_annotation, 'ARA structure ID')  

                        boundary_y = self.annotation_boundary_nii[:, alignment, :]
                        rotated_boundary = np.rot90(boundary_y, k=2)  # 旋转180度
                        transposed_boundary = rotated_boundary.T  # 转置操作
                        ConvNext_boundary = convert_to_png_like_array(transposed_boundary)                      
                        
                    
                        if isinstance(brain_region, str) and brain_region.strip():
                            brain_regions = parse_brain_regions(brain_region)
                            for region in brain_regions:
                                if self.voxel == "nii":
                                    mask_nii_image = sitk.ReadImage(os.path.join(self.mask_nii_folder, f"{region}.nii.gz"))
                                    mask_nii = sitk.GetArrayViewFromImage(mask_nii_image)
                                    mask_nii = np.transpose(mask_nii, (0, 2, 1))[::-1, ::-1, ::-1]
                                elif self.voxel == "nrrd":
                                    mask_nii, _ = nrrd.read(os.path.join(self.mask_nii_folder, f"{region}.nrrd"))
                                    mask_nii = np.transpose(mask_nii, (2, 0, 1))[:, ::-1, ::-1]
                                
                                mask_y = mask_nii[:, alignment, :]
                                rotated_mask = np.rot90(mask_y, k=2)  # 旋转180度
                                transposed_mask = rotated_mask.T  # 转置操作
                                ConvNext_mask_npy = cv2.resize(transposed_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                ConvNext_mask = convert_to_png_like_array(transposed_mask)
                                mask_resized = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                                region_mask_data.append({
                                    "region": region,
                                    "mask_resized": mask_resized,
                                    "ConvNext_mask_npy": ConvNext_mask_npy
                                })
                        else:
                            region_mask_data = []

                        
                    # 调整注释图像的尺寸
                    annotation_resized = cv2.resize(ConvNext_annotation, (256, 256), interpolation=cv2.INTER_AREA)
                    boundary_resized = cv2.resize(ConvNext_boundary, (256, 256), interpolation=cv2.INTER_AREA)
                    
                    # 获取 RGB 图像
                    rgb_image = annotation_resized
                    
                    # 获取对应的 step1_prediction
                    step1_pred = self.step1_predictions[idx]  # 从 step1_predictions 中选择对应的预测图像
                            
                    # 调用 image_affine_with_ConvNext 来生成相应的结果
                    source_test, target_test, in_shape = image_affine_with_ConvNext(step1_pred, ConvNext_template)                
                    
                    
                    
                    # Step 2: Affine Registration    
                    source_ima, source_map = process_with_affine(source_test)
                    target_ima, target_map = process_with_affine(target_test)

                    trans_test = self.model_aff.predict((source_map, target_map))
                    affine_field = trans_test[0]

                    moved_ima = vxm.layers.SpatialTransformer(fill_value=0)((source_ima, trans_test))
                    template = tf.squeeze(moved_ima, axis=0)
                    step2_template = tf.cast(template, dtype=tf.uint8)

                    if save_temp:
                        #step1 result saved
                        step1_output_path = os.path.join(image_output_dirs[idx], f"step1_predicted.png")
                        array_to_img(self.step1_predictions[idx]).save(step1_output_path)  # 保存预测结果
                        #step2 result saved
                        np.save(os.path.join(image_output_dirs[idx], "step2_affine_field.npy"), affine_field)
                        encoded_template = tf.image.encode_png(step2_template)
                        tf.io.write_file(os.path.join(image_output_dirs[idx], "step2_template.png"), encoded_template)

                    # Step2.5 Block & Mask
                    if brain_state != "complete":
                        binary_step1_prediction = (self.step1_predictions[idx] > 15).astype(np.uint8)
                        # 对 binary1 进行膨胀操作（向上下左右扩展 5 像素）
                        gray_mask = binary_step1_prediction[:, :, 0]
                        structure = np.ones((11, 11))  # 5 pixels in all directions → kernel size = 2*5 + 1
                        expanded_mask = binary_dilation(gray_mask, structure=structure).astype(np.uint8)
                        expanded_mask_3d = expanded_mask[..., np.newaxis]
                        # 使用 mask 筛选 binary2 / 原图
                        step2_template = step2_template * expanded_mask_3d  # 如果你想应用到原图像素

                        
                    # Step 3: U-Net Registration
                    moving_image = scale_from_array(step2_template)
                    fixed_image = scale_from_array(self.step1_predictions[idx])

                    moving_image = np.array(moving_image) / 255.0
                    fixed_image = np.array(fixed_image) / 255.0

                    moving_image = moving_image[np.newaxis, ..., np.newaxis]
                    fixed_image = fixed_image[np.newaxis, ..., np.newaxis]

                    val_pred = self.vxm_model.predict((moving_image, fixed_image))
                    displacement_field = val_pred[1][0]

                    if save_temp:
                        np.save(os.path.join(image_output_dirs[idx], "step3_displacement_field.npy"), displacement_field)

                    # Step 4: Combine and Visualize
                    total_displacement_field = affine_field + displacement_field
                    raw_image = cv2.imread(image_files[idx])
                    original_shape = raw_image.shape[:-1]
                    target_shape = (original_shape[1], original_shape[0])

                    resized_fixed_image = cv2.resize(self.step1_predictions[idx], target_shape, interpolation=cv2.INTER_LINEAR)

                    # Moving template
                    new_template = scale_from_array(step2_template)
                    new_template = new_template.astype('float32') / 255.0
                    new_template = new_template[..., np.newaxis]

                    # Moving annotation
                    R_channel = rgb_image[:, :, 0].astype('float32') / 255.0
                    G_channel = rgb_image[:, :, 1].astype('float32') / 255.0
                    B_channel = rgb_image[:, :, 2].astype('float32') / 255.0
            
                    R_channel = R_channel[..., np.newaxis]
                    G_channel = G_channel[..., np.newaxis]
                    B_channel = B_channel[..., np.newaxis]

                    # Moving boundary
                    new_boundary_R = boundary_resized[:, :, 0].astype('float32') / 255.0
                    new_boundary_G = boundary_resized[:, :, 1].astype('float32') / 255.0
                    new_boundary_B = boundary_resized[:, :, 2].astype('float32') / 255.0

                    new_boundary_R = new_boundary_R[..., np.newaxis]
                    new_boundary_G = new_boundary_G[..., np.newaxis]
                    new_boundary_B = new_boundary_B[..., np.newaxis]
            
                    # SpatialTransformer
                    spatial_transformer = vxm.layers.SpatialTransformer()

                    # Template
                    transformed_template = spatial_transformer(
                        [new_template[np.newaxis, ...], displacement_field[np.newaxis, ...]]
                    )
                    transformed_template = transformed_template[0, ..., 0]

                    if brain_state != "complete":
                        mask_transformed_template = (transformed_template.numpy() > 4 / 255).astype(np.uint8)
                        mask_3d = mask_transformed_template[..., np.newaxis]  # shape [H, W, 1]

                    resized_transformed_template = cv2.resize(
                        transformed_template.numpy(), target_shape, interpolation=cv2.INTER_LINEAR
                    )


                    # Annotation
                    transformed_R_channel = spatial_transformer(
                        [R_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    )
                    transformed_G_channel = spatial_transformer(
                        [G_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
                    transformed_B_channel = spatial_transformer(
                        [B_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
                    transformed_R_channel = transformed_R_channel[0, ..., 0] * 255
                    transformed_G_channel = transformed_G_channel[0, ..., 0] * 255
                    transformed_B_channel = transformed_B_channel[0, ..., 0] * 255

                    # Boundary
                    transformed_boundary_R = spatial_transformer(
                        [new_boundary_R[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    )
                    transformed_boundary_G = spatial_transformer(
                        [new_boundary_G[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
                    transformed_boundary_B = spatial_transformer(
                        [new_boundary_B[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
                    ) 
            
                    transformed_boundary_R = transformed_boundary_R[0, ..., 0] * 255
                    transformed_boundary_G = transformed_boundary_G[0, ..., 0] * 255
                    transformed_boundary_B = transformed_boundary_B[0, ..., 0] * 255
                    # Save displacement_field
                    np.save(os.path.join(image_output_dirs[idx], "total_displacement_field.npy"), total_displacement_field)
            
            
            
                    # 保存 resized_transformed_template & resized_transformed_annotation
                    resized_transformed_template_uint8 = (resized_transformed_template * 255).astype(np.uint8)
                    rgb_moved = np.stack((transformed_R_channel, transformed_G_channel, transformed_B_channel), axis=2).astype(np.uint8)
                    if mode == "block" and brain_state != "complete":
                        target_color = np.array([17, 110, 175], dtype=np.uint8)
                        color_layer = np.ones_like(rgb_moved) * target_color
                        result_rgb = rgb_moved * mask_3d + color_layer * (1 - mask_3d)
                        rgb_moved = result_rgb.astype(np.uint8)
                    elif mode == "Allen" and brain_state != "complete":
                        target_color = np.array([0, 0, 0], dtype=np.uint8)
                        color_layer = np.ones_like(rgb_moved) * target_color
                        result_rgb = rgb_moved * mask_3d + color_layer * (1 - mask_3d)
                        rgb_moved = result_rgb.astype(np.uint8)
                    resized_transformed_annotation = cv2.resize(rgb_moved, target_shape, interpolation=cv2.INTER_LINEAR)
                    boundary_moved = np.stack((transformed_boundary_R, transformed_boundary_G, transformed_boundary_B), axis=2).astype(np.uint8)

                    if brain_state != "complete":
                        target_color = np.array([0, 0, 0], dtype=np.uint8)
                        color_layer = np.ones_like(boundary_moved) * target_color  # shape [H, W, 3]
                        result_boundary = boundary_moved * mask_3d + color_layer * (1 - mask_3d)
                        boundary_moved = result_boundary.astype(np.uint8)
                    resized_transformed_boundary = cv2.resize(boundary_moved, target_shape, interpolation=cv2.INTER_LINEAR)
                    raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                    raw_image_diaplay = raw_image_rgb
            
                    # enlarge or not
                    if enlarge:
                        binary_map = np.where(resized_transformed_template_uint8 > 15, 1.0, 0.0).astype(np.uint8)  # 二值化
                        rows, cols = np.where(binary_map == 1)
                        # 确定上下左右边界
                        top = np.min(rows)    # 最小行号 -> 上边界
                        bottom = np.max(rows) # 最大行号 -> 下边界
                        left = np.min(cols)   # 最小列号 -> 左边界
                        right = np.max(cols)  # 最大列号 -> 右边界
                        resized_transformed_template_uint8 = resized_transformed_template_uint8[top:bottom+1, left:right+1]
                        resized_transformed_annotation = resized_transformed_annotation[top:bottom+1, left:right+1]
                        resized_transformed_boundary = resized_transformed_boundary[top:bottom+1, left:right+1]
                        raw_image_rgb = raw_image_rgb[top:bottom+1, left:right+1]
                        raw_image = raw_image[top:bottom+1, left:right+1]

                    # save
                    plt.imsave(os.path.join(image_output_dirs[idx], "resized_transformed_template.png"), resized_transformed_template_uint8, cmap='gray')
                    plt.imsave(os.path.join(image_output_dirs[idx], "resized_transformed_annotation.png"), resized_transformed_annotation)
                    plt.imsave(os.path.join(image_output_dirs[idx], "resized_transformed_boundary.png"), resized_transformed_boundary, cmap='gray')
                    
                    #visualize
                    if mode == "block":
                        overlay = cv2.resize(resized_transformed_annotation, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                        Total_overlay = overlay_images(raw_image_rgb, resized_transformed_annotation, alpha=0.3)
                        result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(image_output_dirs[idx], "Total_brain_registration.png"), result_bgr)

                    if mode == "Allen":
                        overlay = cv2.resize(resized_transformed_annotation, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                        Total_overlay = overlay_images(raw_image_rgb, resized_transformed_annotation, alpha=0.3)
                        result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(image_output_dirs[idx], "Total_brain_registration.png"), result_bgr)

                    if mode == "line":
                        overlay = cv2.resize(resized_transformed_boundary, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                        Total_overlay = overlay_images(raw_image_rgb, resized_transformed_boundary, alpha=0.3)
                        result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(image_output_dirs[idx], "Total_brain_registration.png"), result_bgr)
                        
                    file_name = os.path.basename(image_output_dirs[idx])
                    
                    
                    if  temp_result:         
                        plt.figure(figsize=(18, 5))

                        # Raw Image
                        plt.subplot(1, 4, 1)
                        plt.imshow(raw_image_diaplay)
                        plt.title("Raw Image")
                        plt.axis("off")

                        # CycleGAN Predicted
                        plt.subplot(1, 4, 2)
                        plt.imshow(resized_fixed_image, cmap="gray")
                        plt.title("CycleGAN Predicted")
                        plt.axis("off")

                        # Template Image
                        plt.subplot(1, 4, 3)
                        plt.imshow(resized_transformed_template, cmap="gray")
                        plt.title("Brain Template")
                        plt.axis("off")

                        # Annotation
                        plt.subplot(1, 4, 4)
                        plt.imshow(resized_transformed_annotation)
                        plt.title("Annotation")
                        plt.axis("off")
                        
                        plt.tight_layout()
                        plt.close() 
                        
                        plt.figure(figsize=(12,8))          
                
                        plt.subplot(1, 1, 1)
                        plt.imshow(Total_overlay)
                        plt.title(f"{file_name} registration", fontsize=20)
                        plt.axis("off")    
                        
                        plt.tight_layout()
                        plt.close()     
                        
                    else:
                
                        plt.figure(figsize=(12,8))          
                
                        plt.subplot(1, 1, 1)
                        plt.imshow(Total_overlay)
                        plt.title(f"{file_name} registration", fontsize=20)
                        plt.axis("off") 
                        
                        plt.tight_layout()
                        plt.close()  
                    
                    
                    
                    
                    if region_mask_data:
                        for item in region_mask_data:
                            region = item["region"]
                            mask_resized = item["mask_resized"]
                            ConvNext_mask_npy = item["ConvNext_mask_npy"]

                            #moving brain region
                            mask = scale_from_array(mask_resized)  # 灰度图
                            mask = mask.astype('float32') / 255.0  # 归一化到 [0, 1]
                            mask = mask[..., np.newaxis]  # 添加通道维度，shape: [H, W, 1]
                            mask_npy = ConvNext_mask_npy[..., np.newaxis]

                            transformed_mask = spatial_transformer([mask[np.newaxis, ...], total_displacement_field[np.newaxis, ...]])
                            transformed_mask = transformed_mask[0, ..., 0]  # shape: [H, W]
                            transformed_mask_np = transformed_mask.numpy()
                            resized_transformed_mask = cv2.resize(transformed_mask_np, target_shape, interpolation=cv2.INTER_LINEAR)

                            mask_npy = mask_npy.astype(np.float32)
                            transformed_mask_npy = spatial_transformer([mask_npy[np.newaxis, ...], total_displacement_field[np.newaxis, ...]])
                            transformed_mask_npy = transformed_mask_npy[0, ..., 0]  # shape: [H, W]
                            transformed_mask_npy_np = transformed_mask_npy.numpy()
                            resize_mask_npy = cv2.resize(transformed_mask_npy_np, target_shape, interpolation=cv2.INTER_LINEAR)
                    
                            resized_transformed_mask_uint8 = (resized_transformed_mask * 255).astype(np.uint8)                    
                            resize_mask_npy = (resize_mask_npy > 0.5).astype(np.uint8)

                    
                            if enlarge:
                                resized_transformed_mask_uint8 = resized_transformed_mask_uint8[top:bottom+1, left:right+1]
                                resize_mask_npy = resize_mask_npy[top:bottom+1, left:right+1]
                    
                            # 保存 resized_transformed_mask
                            plt.imsave(os.path.join(image_output_dirs[idx], f"{region}_resized_transformed_mask.png"), resized_transformed_mask_uint8, cmap='gray')  # 灰度图像
                            np.save(os.path.join(image_output_dirs[idx], f"{region}_resized_transformed_mask.npy"), resize_mask_npy)
                
                            # 调用函数，使用黑色阈值去掉接近黑色的边缘
                            if mode == "block" or mode == "Allen":
                                resized_raw_image_uint8 = raw_image.astype(np.uint8)
                                combined_image = overlay_images_with_transparency(
                                background_array=resized_raw_image_uint8,  # 背景图 (H, W, 3)
                                overlay_array=resized_transformed_mask_uint8,  # 叠加图 (H, W)
                                alpha=0.5,  # 透明度
                                black_threshold=100,  # 设置黑色阈值（去掉小于 20 的灰度值）
                                output_path=os.path.join(image_output_dirs[idx], f"{region}_registration.png")
                                )
                            if mode == "line":
                                combined_image = brain_region_contour_overlay(resized_transformed_mask_uint8, raw_image, bgr_color)
                                cv2.imwrite(os.path.join(image_output_dirs[idx], f"{region}_registration.png"), combined_image)
                                combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
                    
                    

                            if temp_result:
                                # Brain Region (mask_resized)
                                plt.figure(figsize=(12,8))
                                
                                plt.subplot(1, 2, 1)
                                plt.imshow(resized_transformed_mask, cmap="gray")
                                plt.title(f"Brain Region: {region}")
                                plt.axis("off")
                        
                                plt.subplot(1, 2, 2)
                                plt.imshow(combined_image)
                                plt.title(f"Brain region registration: {region}", fontsize=20)
                                plt.axis("off")
                        
                                plt.tight_layout()
                                plt.close()  

                            else:
                        
                                plt.figure(figsize=(12,8))             
                                plt.imshow(combined_image)
                                plt.title(f"Brain region registration: {region}", fontsize=20)
                                plt.axis("off")
                        
                                plt.tight_layout()
                                plt.close()  
            
                    gc.collect()        
                
                
   
                
     ############## Normal ############              
                
        
        else:# 如果没有启用 angle_integration，继续执行后续步骤
            # Step 1: Style Transfer
            img_tensor, field = preprocess_raw_image(raw_image_path)
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            if field == "dark_field":
                prediction = self.gen_G_df(img_tensor, training=False)[0].numpy()
            elif field == "bright_field":
                prediction = self.gen_G_bf(img_tensor, training=False)[0].numpy()
            prediction = (prediction + 1.0) / 2.0
            prediction = np.clip(prediction, 0, 1)  # 确保在 [0,1] 之间
            step1_prediction = (prediction * 255).astype(np.uint8)
            step1_prediction = np.repeat(step1_prediction, 3, axis=-1)
            step1_prediction_temp = step1_prediction
            if save_temp:
                step1_output_path = os.path.join(output_dir, "step1_predicted.png")
                array_to_img(step1_prediction).save(step1_output_path)
            
                
            # Step1.5: Layer Registration
            if brain_state == "hemisphere":
                step1_prediction_temp = step1_prediction_temp[:, :, 0]
                step1_prediction_temp, (px1, py1, px2, py2), (cx, cy), hemisphere = process_brain_image_pipeline(step1_prediction_temp)
                step1_prediction_temp = np.repeat(step1_prediction_temp[..., np.newaxis], 3, axis=-1)

            if section_layer is None:
                dataset, filenames = create_image_array_generator_with_customization(
                    image_array=step1_prediction_temp,
                    target_size=(224, 224),
                    batch_size=1,
                    color_mode="rgb",  # 可选 'rgb' 或 'grayscale'
                    preprocessing_function=gray_scale_array,
                )
            
                predictions_df = predictions_util(
                    model=self.ConvNext,
                    dataset=dataset,
                    weights=self.ConvNext_weights,
                    filenames=filenames
                )
       
                alignment = predictions_df.iloc[0, 1:].values.tolist()
                if brain_state == "hemisphere":
                    alignment_df = pd.DataFrame({
                        'alignment': [alignment],
                        'px1': [px1],
                        'py1': [py1],
                        'px2': [px2],
                        'py2': [py2],
                        'cx': [cx],
                        'cy': [cy],
                        'hemisphere': [hemisphere]
                    })
                else:
                    alignment_df = pd.DataFrame({'alignment': [alignment]})
                alignment_df.to_csv(os.path.join(output_dir, "alignment_df.csv"))

                if self.voxel == "nii":
                    ConvNext_template = generate_target_slice(alignment, self.volume_nii)
                    ConvNext_annotation = generate_target_slice(alignment, self.annotation_nii)
                elif self.voxel == "nrrd":
                    ConvNext_template = generate_target_slice_voxel25(alignment, self.volume_nii)
                    ConvNext_annotation = generate_target_slice_voxel25(alignment, self.annotation_nii)

                ConvNext_template = convert_to_png_like_array(ConvNext_template)
                ConvNext_template = cv2.resize(ConvNext_template, (256, 256), interpolation=cv2.INTER_AREA)
                plt.figure(figsize=(18, 5))
                plt.imshow(ConvNext_template)
                plt.title("Raw Image")
                plt.axis("off")

                
                if mode in ("block", "line"):
                    ConvNext_annotation = convert_to_png_like_array_with_tab20(ConvNext_annotation)
                if mode == "Allen":
                    if self.voxel == "nii":
                        ConvNext_annotation = colorize(self.channels, ConvNext_annotation, 'substructure')
                    elif self.voxel == "nrrd":
                        ConvNext_annotation = colorize(self.channels, ConvNext_annotation, 'ARA structure ID')
                ConvNext_boundary = generate_target_slice(alignment, self.annotation_boundary_nii)
                ConvNext_boundary = convert_to_png_like_array(ConvNext_boundary)
                annotation_resized = cv2.resize(ConvNext_annotation, (256, 256), interpolation=cv2.INTER_AREA)
                boundary_resized = cv2.resize(ConvNext_boundary, (256, 256), interpolation=cv2.INTER_AREA)
                rgb_image = annotation_resized

                if brain_state == "hemisphere":
                    ConvNext_template = crop_half_by_centroid_and_axis(ConvNext_template, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0))  # 黑色填充
                    if mode in ("block", "line"):
                        rgb_image = crop_half_by_centroid_and_axis(rgb_image, px1, py1, px2, py2, cx, cy, fill_value=(17, 110, 175))  # 蓝色填充
                    if mode == "Allen":
                        rgb_image = crop_half_by_centroid_and_axis(rgb_image, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0))  # 黑色填充
                    boundary_resized = crop_half_by_centroid_and_axis(boundary_resized, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0))  # 黑色填充

                source_test, target_test, in_shape = image_affine_with_ConvNext(
                step1_prediction, ConvNext_template
                )
            
                if isinstance(brain_region, str) and brain_region.strip():
                    brain_regions = parse_brain_regions(brain_region)
                    for region in brain_regions:
                        if self.voxel == "nii":
                            mask_nii_image = sitk.ReadImage(os.path.join(self.mask_nii_folder, f"{region}.nii.gz"))
                            mask_nii = sitk.GetArrayViewFromImage(mask_nii_image)
                            mask_nii = np.transpose(mask_nii, (0, 2, 1))[::-1, ::-1, ::-1]
                            ConvNext_mask = generate_target_slice(alignment, mask_nii)
                        elif self.voxel == "nrrd":
                            mask_nii, _ = nrrd.read(os.path.join(self.mask_nii_folder, f"{region}.nrrd"))
                            mask_nii = np.transpose(mask_nii, (2, 0, 1))[:, ::-1, ::-1]
                            ConvNext_mask = generate_target_slice_voxel25(alignment, mask_nii)
                        ConvNext_mask_npy = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                        ConvNext_mask = convert_to_png_like_array(ConvNext_mask)
                        mask_resized = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                        if brain_state == "hemisphere":
                            mask_resized = crop_half_by_centroid_and_axis(mask_resized, px1, py1, px2, py2, cx, cy, fill_value=(0, 0, 0))
                        region_mask_data.append({
                            "region": region,
                            "mask_resized": mask_resized,
                            "ConvNext_mask_npy": ConvNext_mask_npy
                        })
                else:
                    region_mask_data = []

                
            
            elif isinstance(section_layer, str):

                filtered_df = self.df[self.df['Layer'].astype(str) == section_layer]
                subset_df = filtered_df.iloc[:, 0:7]
                    
                # 提取每一行的 alignment 信息
                alignment = subset_df['slice'].iloc[0]
                if brain_state == "hemisphere":
                    alignment_df = pd.DataFrame({
                        'alignment': [alignment],
                        'px1': [px1],
                        'py1': [py1],
                        'px2': [px2],
                        'py2': [py2],
                        'cx': [cx],
                        'cy': [cy],
                        'hemisphere': [hemisphere]
                    })
                else:
                    alignment_df = pd.DataFrame({'alignment': [alignment]})
                alignment_df.to_csv(os.path.join(output_dir, "alignment_df.csv"))


                # 生成对应的 ConvNext_template 和 ConvNext_annotation
                slice_y = self.volume_nii[:, alignment, :]
                rotated_slice = np.rot90(slice_y, k=2)  # 旋转180度
                transposed_slice = rotated_slice.T  # 转置操作
                ConvNext_template = convert_to_png_like_array(transposed_slice)
                ConvNext_template = cv2.resize(ConvNext_template, (256, 256), interpolation=cv2.INTER_AREA)  
                annotation_y = self.annotation_nii[:, alignment, :]
                rotated_annotation = np.rot90(annotation_y, k=2)  # 旋转180度
                transposed_annotation = rotated_annotation.T  # 转置操作
                if mode == "block": 
                    ConvNext_annotation = convert_to_png_like_array_with_tab20(transposed_annotation)
                if mode == "Allen":
                    if self.voxel == "nii":
                        ConvNext_annotation = colorize(self.channels, transposed_annotation, 'substructure')
                    if self.voxel == "nrrd":
                        ConvNext_annotation = colorize(self.channels, transposed_annotation, 'ARA structure ID')
                annotation_resized = cv2.resize(ConvNext_annotation, (256, 256), interpolation=cv2.INTER_AREA)
                rgb_image = annotation_resized
                boundary_y = self.annotation_boundary_nii[:, alignment, :]
                rotated_boundary = np.rot90(boundary_y, k=2)  # 旋转180度
                transposed_boundary = rotated_boundary.T  # 转置操作
                ConvNext_boundary = convert_to_png_like_array(transposed_boundary)
                boundary_resized = cv2.resize(ConvNext_boundary, (256, 256), interpolation=cv2.INTER_AREA)

                if brain_state == "hemisphere":
                    ConvNext_template = mask_half_image(ConvNext_template, side= hemisphere, fill_color=(0, 0, 0))
                    if mode == "block":
                        rgb_image = mask_half_image(rgb_image, side= hemisphere, fill_color=(17, 110, 175))
                    if mode == "Allen":
                        rgb_image = mask_half_image(rgb_image, side= hemisphere, fill_color=(0, 0, 0))
                    boundary_resized = mask_half_image(boundary_resized, side= hemisphere, fill_color=(0, 0, 0))
                    
                if isinstance(brain_region, str) and brain_region.strip():
                    brain_regions = parse_brain_regions(brain_region)
                    for region in brain_regions:
                        if self.voxel == "nii":
                            mask_nii_image = sitk.ReadImage(os.path.join(self.mask_nii_folder, f"{brain_region}.nii.gz"))
                            mask_nii = sitk.GetArrayViewFromImage(mask_nii_image)
                            mask_nii = np.transpose(mask_nii, (0, 2, 1))[::-1, ::-1, ::-1]
                        if self.voxel == "nrrd":
                            mask_nii, _ = nrrd.read(os.path.join(self.mask_nii_folder, f"{brain_region}.nrrd"))
                            mask_nii = np.transpose(mask_nii, (2, 0, 1))[:, ::-1, ::-1]

                        mask_y = mask_nii[:, alignment, :]
                        rotated_mask = np.rot90(mask_y, k=2)  # 旋转180度
                        transposed_mask = rotated_mask.T  # 转置操作
                        ConvNext_mask_npy = cv2.resize(transposed_mask, (256, 256), interpolation=cv2.INTER_AREA)
                        ConvNext_mask = convert_to_png_like_array(transposed_mask)
                        mask_resized = cv2.resize(ConvNext_mask, (256, 256), interpolation=cv2.INTER_AREA)
                        if brain_state == "hemisphere":
                            mask_resized = mask_half_image(mask_resized, side= hemisphere, fill_color=(0, 0, 0))
                        region_mask_data.append({
                            "region": region,
                            "mask_resized": mask_resized,
                            "ConvNext_mask_npy": ConvNext_mask_npy
                        })
                else:
                    region_mask_data = []
                source_test, target_test, in_shape = image_affine_with_ConvNext(
                step1_prediction, ConvNext_template
                )

            else:
                raise ValueError("Invalid section_layer! Must be an str or None.")
        
            # Step 2: Affine Registration    
            source_ima, source_map = process_with_affine(source_test)
            target_ima, target_map = process_with_affine(target_test)

            trans_test = self.model_aff.predict((source_map, target_map))
            affine_field = trans_test[0]

            moved_ima = vxm.layers.SpatialTransformer(fill_value=0)((source_ima, trans_test))
            template = tf.squeeze(moved_ima, axis=0)
            step2_template = tf.cast(template, dtype=tf.uint8)

            if save_temp:
                np.save(os.path.join(output_dir, "step2_affine_field.npy"), affine_field)
                encoded_template = tf.image.encode_png(step2_template)
                tf.io.write_file(os.path.join(output_dir, "step2_template.png"), encoded_template)

            # Step2.5 Block & Mask
            if brain_state != "complete":
                binary_step1_prediction = (step1_prediction > 15).astype(np.uint8)
                # 对 binary1 进行膨胀操作（向上下左右扩展 5 像素）
                gray_mask = binary_step1_prediction[:, :, 0]
                structure = np.ones((11, 11))  # 5 pixels in all directions → kernel size = 2*5 + 1
                expanded_mask = binary_dilation(gray_mask, structure=structure).astype(np.uint8)
                expanded_mask_3d = expanded_mask[..., np.newaxis]
                # 使用 mask 筛选 binary2 / 原图
                step2_template = step2_template * expanded_mask_3d  # 如果你想应用到原图像素
            
            # Step 3: U-Net Registration
            moving_image = scale_from_array(step2_template)
            fixed_image = scale_from_array(step1_prediction)

            moving_image = np.array(moving_image) / 255.0
            fixed_image = np.array(fixed_image) / 255.0

            moving_image = moving_image[np.newaxis, ..., np.newaxis]
            fixed_image = fixed_image[np.newaxis, ..., np.newaxis]

            val_pred = self.vxm_model.predict((moving_image, fixed_image))
            displacement_field = val_pred[1][0]

            if save_temp:
                np.save(os.path.join(output_dir, "step3_displacement_field.npy"), displacement_field)

            # Step 4: Combine and Visualize
            total_displacement_field = affine_field + displacement_field
            raw_image = cv2.imread(raw_image_path)
            original_shape = raw_image.shape[:-1]
            target_shape = (original_shape[1], original_shape[0])

            resized_fixed_image = cv2.resize(step1_prediction, target_shape, interpolation=cv2.INTER_LINEAR)

            # Moving template
            new_template = scale_from_array(step2_template)
            new_template = new_template.astype('float32') / 255.0
            new_template = new_template[..., np.newaxis]

            # Moving annotation
            R_channel = rgb_image[:, :, 0].astype('float32') / 255.0
            G_channel = rgb_image[:, :, 1].astype('float32') / 255.0
            B_channel = rgb_image[:, :, 2].astype('float32') / 255.0
        
            R_channel = R_channel[..., np.newaxis]
            G_channel = G_channel[..., np.newaxis]
            B_channel = B_channel[..., np.newaxis]

            # Moving boundary
            new_boundary_R = boundary_resized[:, :, 0].astype('float32') / 255.0
            new_boundary_G = boundary_resized[:, :, 1].astype('float32') / 255.0
            new_boundary_B = boundary_resized[:, :, 2].astype('float32') / 255.0

            new_boundary_R = new_boundary_R[..., np.newaxis]
            new_boundary_G = new_boundary_G[..., np.newaxis]
            new_boundary_B = new_boundary_B[..., np.newaxis]

            # SpatialTransformer
            spatial_transformer = vxm.layers.SpatialTransformer()

            # Template
            transformed_template = spatial_transformer(
                [new_template[np.newaxis, ...], displacement_field[np.newaxis, ...]]
            )
            transformed_template = transformed_template[0, ..., 0]
            if brain_state != "complete":
                mask_transformed_template = (transformed_template.numpy() > 4 / 255).astype(np.uint8)
                mask_3d = mask_transformed_template[..., np.newaxis]  # shape [H, W, 1]

            resized_transformed_template = cv2.resize(
                transformed_template.numpy(), target_shape, interpolation=cv2.INTER_LINEAR
            )

            # Annotation
            transformed_R_channel = spatial_transformer(
                [R_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
            )
            transformed_G_channel = spatial_transformer(
                [G_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
            ) 
            transformed_B_channel = spatial_transformer(
                [B_channel[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
            ) 
        
            transformed_R_channel = transformed_R_channel[0, ..., 0] * 255
            transformed_G_channel = transformed_G_channel[0, ..., 0] * 255
            transformed_B_channel = transformed_B_channel[0, ..., 0] * 255
        
        
            # Boundary
            transformed_boundary_R = spatial_transformer(
                [new_boundary_R[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
            )
            transformed_boundary_G = spatial_transformer(
                [new_boundary_G[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
            ) 
            transformed_boundary_B = spatial_transformer(
                [new_boundary_B[np.newaxis, ...], total_displacement_field[np.newaxis, ...]]
            ) 
        
            transformed_boundary_R = transformed_boundary_R[0, ..., 0] * 255
            transformed_boundary_G = transformed_boundary_G[0, ..., 0] * 255
            transformed_boundary_B = transformed_boundary_B[0, ..., 0] * 255

            # Save displacement_field
            np.save(os.path.join(output_dir, "total_displacement_field.npy"), total_displacement_field)
        
        
        
            # 保存 resized_transformed_template & resized_transformed_annotation
            resized_transformed_template_uint8 = (resized_transformed_template * 255).astype(np.uint8)
            rgb_moved = np.stack((transformed_R_channel, transformed_G_channel, transformed_B_channel), axis=2).astype(np.uint8)
            if brain_state != "complete":
                if mode == "block":
                    target_color = np.array([17, 110, 175], dtype=np.uint8)
                    color_layer = np.ones_like(rgb_moved) * target_color
                    result_rgb = rgb_moved * mask_3d + color_layer * (1 - mask_3d)
                    rgb_moved = result_rgb.astype(np.uint8)
                elif mode == "Allen":
                    target_color = np.array([0, 0, 0], dtype=np.uint8)
                    color_layer = np.ones_like(rgb_moved) * target_color
                    result_rgb = rgb_moved * mask_3d + color_layer * (1 - mask_3d)
                    rgb_moved = result_rgb.astype(np.uint8)


            resized_transformed_annotation = cv2.resize(rgb_moved, target_shape, interpolation=cv2.INTER_LINEAR)
            boundary_moved = np.stack((transformed_boundary_R, transformed_boundary_G, transformed_boundary_B), axis=2).astype(np.uint8)
            target_color = np.array([0, 0, 0], dtype=np.uint8)
            color_layer = np.ones_like(boundary_moved) * target_color  # shape [H, W, 3]
            if brain_state != "complete":
                result_boundary = boundary_moved * mask_3d + color_layer * (1 - mask_3d)
                boundary_moved = result_boundary.astype(np.uint8)

            resized_transformed_boundary = cv2.resize(boundary_moved, target_shape, interpolation=cv2.INTER_LINEAR)
            raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            raw_image_diaplay = raw_image_rgb
        
            # enlarge or not
            if enlarge:
                binary_map = np.where(resized_transformed_template_uint8 > 15, 1.0, 0.0).astype(np.uint8)  # 二值化
                rows, cols = np.where(binary_map == 1)
                # 确定上下左右边界
                top = np.min(rows)    # 最小行号 -> 上边界
                bottom = np.max(rows) # 最大行号 -> 下边界
                left = np.min(cols)   # 最小列号 -> 左边界
                right = np.max(cols)  # 最大列号 -> 右边界
                resized_transformed_template_uint8 = resized_transformed_template_uint8[top:bottom+1, left:right+1]
                resized_transformed_annotation = resized_transformed_annotation[top:bottom+1, left:right+1]
                resized_transformed_boundary = resized_transformed_boundary[top:bottom+1, left:right+1]
                raw_image_rgb = raw_image_rgb[top:bottom+1, left:right+1]
                raw_image = raw_image[top:bottom+1, left:right+1]

            # save
            plt.imsave(os.path.join(output_dir, "resized_transformed_template.png"), resized_transformed_template_uint8, cmap='gray')
            plt.imsave(os.path.join(output_dir, "resized_transformed_annotation.png"), resized_transformed_annotation)
            plt.imsave(os.path.join(output_dir, "resized_transformed_boundary.png"), resized_transformed_boundary)
        

            #visualize
            if mode == "block":
                overlay = cv2.resize(resized_transformed_annotation, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                Total_overlay = overlay_images(raw_image_rgb, resized_transformed_annotation, alpha=0.3)
                result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, "Total_brain_registration.png"), result_bgr)

            if mode == "Allen":
                overlay = cv2.resize(resized_transformed_annotation, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                Total_overlay = overlay_images(raw_image_rgb, resized_transformed_annotation, alpha=0.3)
                result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, "Total_brain_registration.png"), result_bgr)            

            if mode == "line":
                overlay = cv2.resize(resized_transformed_boundary, (raw_image_rgb.shape[1], raw_image_rgb.shape[0]))
                Total_overlay = overlay_images(raw_image_rgb, resized_transformed_boundary, alpha=0.3)
                result_bgr = cv2.cvtColor(Total_overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, "Total_brain_registration.png"), result_bgr)


            if  temp_result:         
                plt.figure(figsize=(18, 5))

                # Raw Image
                plt.subplot(1, 4, 1)
                plt.imshow(raw_image_diaplay)
                plt.title("Raw Image")
                plt.axis("off")

                # CycleGAN Predicted
                plt.subplot(1, 4, 2)
                plt.imshow(resized_fixed_image, cmap="gray")
                plt.title("CycleGAN Predicted")
                plt.axis("off")

                # Template Image
                plt.subplot(1, 4, 3)
                plt.imshow(resized_transformed_template, cmap="gray")
                plt.title("Brain Template")
                plt.axis("off")

                # Annotation
                plt.subplot(1, 4, 4)
                plt.imshow(resized_transformed_annotation)
                plt.title("Annotation")
                plt.axis("off")
                        
                plt.tight_layout()
                plt.close() 
                        
                plt.figure(figsize=(12,8))          
                
                plt.subplot(1, 1, 1)
                plt.imshow(Total_overlay)
                plt.title(f"All brain registration", fontsize=20)
                plt.axis("off")    
                        
                plt.tight_layout()
                plt.close()     
                        
            else:
                
                plt.figure(figsize=(12,8))          
                
                plt.subplot(1, 1, 1)
                plt.imshow(Total_overlay)
                plt.title(f"All brain registration", fontsize=20)
                plt.axis("off") 
                        
                plt.tight_layout()
                plt.close()  


            if region_mask_data:
                for item in region_mask_data:
                    region = item["region"]
                    mask_resized = item["mask_resized"]
                    ConvNext_mask_npy = item["ConvNext_mask_npy"]

                    #moving brain region
                    mask = scale_from_array(mask_resized)  # 灰度图
                    mask = mask.astype('float32') / 255.0  # 归一化到 [0, 1]
                    mask = mask[..., np.newaxis]  # 添加通道维度，shape: [H, W, 1]
                    mask_npy = ConvNext_mask_npy[..., np.newaxis]

                    transformed_mask = spatial_transformer([mask[np.newaxis, ...], total_displacement_field[np.newaxis, ...]])
                    transformed_mask = transformed_mask[0, ..., 0]  # shape: [H, W]
                    transformed_mask_np = transformed_mask.numpy()

                    mask_npy = mask_npy.astype(np.float32)
                    transformed_mask_npy = spatial_transformer([mask_npy[np.newaxis, ...], total_displacement_field[np.newaxis, ...]])
                    transformed_mask_npy = transformed_mask_npy[0, ..., 0]  # shape: [H, W]
                    transformed_mask_npy_np = transformed_mask_npy.numpy()

                    resized_transformed_mask = cv2.resize(transformed_mask_np, target_shape, interpolation=cv2.INTER_LINEAR)

                    resize_mask_npy = cv2.resize(transformed_mask_npy_np, target_shape, interpolation=cv2.INTER_LINEAR)
                    resize_mask_npy = (resize_mask_npy > 0.5).astype(np.uint8)
                
                    resized_transformed_mask_uint8 = (resized_transformed_mask * 255).astype(np.uint8)
                
                    if enlarge:
                        resized_transformed_mask_uint8 = resized_transformed_mask_uint8[top:bottom+1, left:right+1]
                        resize_mask_npy = resize_mask_npy[top:bottom+1, left:right+1]
                
                    # 保存 resized_transformed_mask
                    plt.imsave(os.path.join(output_dir, f"{region}_resized_transformed_mask.png"), resized_transformed_mask_uint8, cmap='gray')  # 灰度图像
                    np.save(os.path.join(output_dir, f"{region}_resized_transformed_mask.npy"), resize_mask_npy)
            
                    # 调用函数，使用黑色阈值去掉接近黑色的边缘
                    if mode == "block" or mode == "Allen":
                        resized_raw_image_uint8 = raw_image.astype(np.uint8)
                        combined_image = overlay_images_with_transparency(
                        background_array=resized_raw_image_uint8,  # 背景图 (H, W, 3)
                        overlay_array=resized_transformed_mask_uint8,  # 叠加图 (H, W)
                        alpha=0.5,  # 透明度
                        black_threshold=100,  # 设置黑色阈值（去掉小于 20 的灰度值）
                        output_path=os.path.join(output_dir, f"{region}_registration.png")
                        )
                    if mode == "line":
                        combined_image = brain_region_contour_overlay(resized_transformed_mask_uint8, raw_image, bgr_color)
                        cv2.imwrite(os.path.join(output_dir, f"{region}_registration.png"), combined_image)
                        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
          
                    if temp_result:
                        # Brain Region (mask_resized)
                        plt.figure(figsize=(12,8))
                                
                        plt.subplot(1, 2, 1)
                        plt.imshow(resized_transformed_mask, cmap="gray")
                        plt.title(f"Brain Region: {region}")
                        plt.axis("off")
                        
                        plt.subplot(1, 2, 2)
                        plt.imshow(combined_image)
                        plt.title(f"Brain region registration: {region}", fontsize=20)
                        plt.axis("off")
                        
                        plt.tight_layout()
                        plt.close()  

                    else:
                        
                        plt.figure(figsize=(12,8))             
                        plt.imshow(combined_image)
                        plt.title(f"Brain region registration: {region}", fontsize=20)
                        plt.axis("off")
                        
                        plt.tight_layout()
                        plt.close()  

            
            gc.collect()