import SimpleITK as sitk
import nibabel as nib
import nrrd
import os
import tensorflow as tf
import voxelmorph as vxm
import numpy as np
import pandas as pd
from .models.genG import get_resnet_generator
from .models.ConvNext import initialise_network

def initialize_models(voxel: str = 'nii'):
    
    module_dir = os.path.dirname(os.path.abspath(__file__))


    if voxel == 'nii':
        #nii meta paths
        nrrd_meta_path = os.path.join(module_dir, "metadata", "Allen_nii", "template_meta.csv")
        df = pd.read_csv(nrrd_meta_path, index_col = 0)
        
        #nii paths
        template_nii_path = os.path.join(module_dir, "metadata", "Allen_nii", "average_template_10.nii.gz")  # Allen_template_10.nii
        annotation_nii_path = os.path.join(module_dir, "metadata", "Allen_nii", "annotation_10.nii.gz")  # Allen_annotation_10.nii
        annotation_boundary_nii_path = os.path.join(module_dir, "metadata", "Allen_nii", "annotation_boundary_10.nii.gz")  # Allen_boundary_10.nii
        mask_nii_folder = os.path.join(module_dir, "metadata", "Allen_nii", "mask")  # Allen_Mask_10_folder
        
        #nii Load
        volume_nii = nib.load(template_nii_path)
        volume_nii = volume_nii.get_fdata()
        volume_nii = np.transpose(volume_nii, (0, 2, 1))[::-1, ::-1, ::-1]

        annotation_nii = nib.load(annotation_nii_path)
        annotation_nii = annotation_nii.get_fdata()
        annotation_nii = np.transpose(annotation_nii, (0, 2, 1))[::-1, ::-1, ::-1]
        
        annotation_boundary_nii = nib.load(annotation_boundary_nii_path)
        annotation_boundary_nii = annotation_boundary_nii.get_fdata()
        annotation_boundary_nii = np.transpose(annotation_boundary_nii, (0, 2, 1))[::-1, ::-1, ::-1]

        #nii Allen color
        parcellation_annotation_path = os.path.join(module_dir, "metadata", "Allen_color", "parcellation_annotation.csv")  # Allen_template_10.nii
        parcellation_term_set_path = os.path.join(module_dir, "metadata", "Allen_color", "parcellation_term_set.csv")  # Allen_annotation_10.nii
        parcellation_annotation = pd.read_csv(parcellation_annotation_path, index_col = 0)
        parcellation_term_set = pd.read_csv(parcellation_term_set_path, index_col = 0)
        color = parcellation_annotation.groupby(['parcellation_index', 'parcellation_term_set_name'])['color_hex_triplet'].first().unstack()
        color = color[parcellation_term_set['name']] # order columns
        color.columns = ['%s_color' % x for x in color.columns]
        channels = {}
        for c in ['red', 'blue', 'green'] :
            df_color = parcellation_annotation.groupby(['parcellation_index', 'parcellation_term_set_name'])[c].first().unstack()
            df_color = df_color[parcellation_term_set['name']] # order columns
            df_color.columns = ['%s' % (x) for x in color.columns]
            channels[c] = df_color


    elif voxel == 'nrrd':
        #nrrd meta paths
        nrrd_meta_path = os.path.join(module_dir, "metadata", "Allen_nrrd", "template_meta.csv")
        df = pd.read_csv(nrrd_meta_path, index_col = 0)

        #nrrd paths
        template_nii_path = os.path.join(module_dir, "metadata", "Allen_nrrd", "average_template_25.nrrd")  # Allen_template_25.nrrd
        annotation_nii_path = os.path.join(module_dir, "metadata", "Allen_nrrd", "Allen_annotation_25.nrrd")  # Allen_annotation_25.nrrd
        annotation_boundary_nii_path = os.path.join(module_dir, "metadata", "Allen_nii", "annotation_boundary_10.nii.gz")  # Allen_boundary_10.nii
        mask_nii_folder = os.path.join(module_dir, "metadata", "Allen_nrrd", "mask")  # Allen_Mask_25_folder

        #nrrd Load
        volume_nii, _ = nrrd.read(template_nii_path)
        volume_nii = np.transpose(volume_nii, (2, 0, 1))[:, ::-1, ::-1]

        annotation_nii, _ = nrrd.read(annotation_nii_path)
        annotation_nii = np.transpose(annotation_nii, (2, 0, 1))[:, ::-1, ::-1]
        
        annotation_boundary_nii = nib.load(annotation_boundary_nii_path)
        annotation_boundary_nii = annotation_boundary_nii.get_fdata()
        annotation_boundary_nii = np.transpose(annotation_boundary_nii, (2, 0, 1))[:, ::-1, ::-1]


        #nrrd Allen color
        nrrd_color_path = os.path.join(module_dir, "metadata", "Allen_color", "nrrd_with_colors.csv")
        parcellation_annotation = pd.read_csv(nrrd_color_path, index_col=0)

        color_df = parcellation_annotation.groupby('ARA structure ID')[
            ['color_hex_triplet', 'red', 'green', 'blue']
        ].first().reset_index()

        channels = {
            'red': color_df.set_index('ARA structure ID')['red'].rename_axis('ARA structure ID_color'),
            'green': color_df.set_index('ARA structure ID')['green'].rename_axis('ARA structure ID_color'),
            'blue': color_df.set_index('ARA structure ID')['blue'].rename_axis('ARA structure ID_color')
        }
    
    else:
        raise ValueError(f"Unknown voxel format: {voxel}. Use 'nii' or 'nrrd'.")


    # Weights paths   
    #GEN weights
    step1_GEN_weights_bf = os.path.join(module_dir, "metadata", "weights", "genG_weight", "genG_bright_field_1_7.weights.h5")
    step1_GEN_weights_df = os.path.join(module_dir, "metadata", "weights", "genG_weight", "genG_dark_field_1_9.weights.h5")
    #ConvNext_weights
    ConvNext_weights = os.path.join(module_dir, "metadata", "weights", "ConvNext_weight","ConvNext.weights.h5")
    #affine_weights
    step2_affine_weights = os.path.join(module_dir, "metadata", "weights", "affine_weight", "model_aff0.7-0.3.weights.h5")
    #Unet_weights
    step3_Unet_weights = os.path.join(module_dir, "metadata", "weights", "vxm_weight", "vxm_model.weights.h5")

    # Initialize models
    # Step 1
    gen_G_bf = get_resnet_generator(name="generator_G")
    gen_G_df = get_resnet_generator(name="generator_G")
    gen_G_bf.load_weights(step1_GEN_weights_bf)
    gen_G_df.load_weights(step1_GEN_weights_df)
    
    #ConvNext
    ConvNext = initialise_network(ConvNext_weights)

    # Step 2
    model_aff = vxm.networks.VxmAffineFeatureDetector((256, 256))
    model_aff.load_weights(step2_affine_weights)

    # Step 3
    vxm_model = vxm.networks.VxmDense(
        (256, 256),
        [[32, 32, 32, 32], [32, 32, 32, 32, 32, 16]],
        int_steps=0
    )
    vxm_model.load_weights(step3_Unet_weights)

    return df, volume_nii, annotation_nii, annotation_boundary_nii, channels, gen_G_bf, gen_G_df, ConvNext, ConvNext_weights, mask_nii_folder, model_aff, vxm_model