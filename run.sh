export GPU="0"
export CAMERA="SIMPLE_RADIAL"
export EXP_NAME_1="stage1"
export EXP_NAME_2="stage2"
export EXP_NAME_3="stage3"
export EXP_PATH_1=$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1

#
# Ensure that the following environment variables are accessible to the script:
# PROJECT_DIR and DATA_PATH 
#

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

#################
# PREPROCESSING #
#################

# Arrange raw images into a 3D Gaussian Splatting format
echo -e "\e[1;31;43mArranging raw images into 3D Gaussian Splatting format\e[0m"
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python preprocess_raw_images.py \
    --data_path $DATA_PATH || exit 1

# # Run COLMAP reconstruction and undistort the images and cameras
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python convert.py -s $DATA_PATH \
#     --camera $CAMERA --max_size 2400 || exit 1

# Run Matte-Anything
echo -e "\e[1;31;43mRunning Matte-Anything for hair segmentation\e[0m"
conda deactivate && conda activate matte_anything
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_masks.py \
    --data_path $DATA_PATH --image_format png --max_size 2048 || exit 1

# # Filter images using their IQA scores
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python filter_extra_images.py \
#     --data_path $DATA_PATH --max_imgs 128 || exit 1

# Resize images
echo -e "\e[1;31;43mResizing images\e[0m"
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python resize_images.py --data_path $DATA_PATH || exit 1

# Calculate orientation maps
echo -e "\e[1;31;43mCalculating orientation maps\e[0m"
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
    --img_path $DATA_PATH/images_2 \
    --mask_path $DATA_PATH/masks_2/hair \
    --orient_dir $DATA_PATH/orientations_2/angles \
    --conf_dir $DATA_PATH/orientations_2/vars \
    --filtered_img_dir $DATA_PATH/orientations_2/filtered_imgs \
    --vis_img_dir $DATA_PATH/orientations_2/vis_imgs || exit 1

# Run OpenPose
echo -e "\e[1;31;43mRunning OpenPose for pose estimation\e[0m"
conda deactivate && cd $PROJECT_DIR/ext/openpose
mkdir $DATA_PATH/openpose
CUDA_VISIBLE_DEVICES="$GPU" ./build/examples/openpose/openpose.bin \
    --image_dir $DATA_PATH/images_4 \
    --scale_number 4 --scale_gap 0.25 --face --hand --display 0 \
    --write_json $DATA_PATH/openpose/json \
    --write_images $DATA_PATH/openpose/images --write_images_format jpg || exit 1

# Run Face-Alignment
echo -e "\e[1;31;43mRunning Face-Alignment for face keypoint estimation\e[0m"
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_face_alignment.py \
    --data_path $DATA_PATH --image_dir "images_4" || exit 1

# Run PIXIE
echo -e "\e[1;31;43mRunning PIXIE for parametric face fitting\e[0m"
conda deactivate && conda activate pixie-env
cd $PROJECT_DIR/ext/PIXIE
CUDA_VISIBLE_DEVICES="$GPU" python demos/demo_fit_face.py \
    -i $DATA_PATH/images_4 -s $DATA_PATH/pixie \
    --saveParam True --lightTex False --useTex False \
    --rasterizer_type pytorch3d || exit 1

# Merge all PIXIE predictions in a single file
echo -e "\e[1;31;43mMerging all PIXIE predictions in a single file\e[0m"
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python merge_smplx_predictions.py \
    --data_path $DATA_PATH || exit 1

# Convert COLMAP cameras to txt
echo -e "\e[1;31;43mConverting COLMAP cameras to txt format\e[0m"
conda deactivate && conda activate gaussian_splatting_hair
mkdir -p $DATA_PATH/sparse_txt
CUDA_VISIBLE_DEVICES="$GPU" colmap model_converter \
    --input_path $DATA_PATH/sparse/0  \
    --output_path $DATA_PATH/sparse_txt --output_type TXT || exit 1

# Convert COLMAP cameras to H3DS format
echo -e "\e[1;31;43mConverting COLMAP cameras to H3DS format\e[0m"
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python colmap_parsing.py \
    --path_to_scene $DATA_PATH || exit 1

# # Remove raw files to preserve disk space
# rm -rf $DATA_PATH/input $DATA_PATH/images $DATA_PATH/masks $DATA_PATH/iqa*

##################
# RECONSTRUCTION #
##################

# Run 3D Gaussian Splatting reconstruction
echo -e "\e[1;31;43mRunning 3D Gaussian Splatting reconstruction\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python train_gaussians.py \
    -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
    --trainable_cameras --trainable_intrinsics --use_barf \
    --lambda_dorient 0.1 || exit 1

# Skip FLAME LBFGS fit; align the FLAME mean-shape template to an externally
# fitted head mesh (ICTFaceKit topology). Set the two paths below to your fit.
#   EXTERNAL_HEAD_MESH: head_fit.obj from prep5_fit_head.py (must live in the
#                       same world frame as $EXP_PATH_1/cameras/30000_matrices.pkl)
#   ICTFACEKIT_NPZ:     ictfacekit_14062.npz with idx_to_landmark_verts (68 dlib pts)
: "${EXTERNAL_HEAD_MESH:=$DATA_PATH/head_fit_in_colmap.obj}"
: "${ICTFACEKIT_NPZ:=/data0/optistrands/pretrained/ictfacekit_14062.npz}"

echo -e "\e[1;31;43mAligning FLAME template to external head fit\e[0m"
conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python fit_flame_from_external_mesh.py \
    --data_path "$DATA_PATH" \
    --exp_name "$EXP_NAME_1" \
    --external_head_mesh "$EXTERNAL_HEAD_MESH" \
    --ictfacekit_npz "$ICTFACEKIT_NPZ" || exit 1

# Crop the reconstructed scene
echo -e "\e[1;31;43mCropping the reconstructed scene\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python scale_scene_into_sphere.py \
    --path_to_data $DATA_PATH \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000 || exit 1

# Remove hair Gaussians that intersect with the FLAME head mesh
echo -e "\e[1;31;43mRemoving hair Gaussians that intersect with the FLAME head mesh\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python filter_flame_intersections.py \
    --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000 \
    --project_dir $PROJECT_DIR/ext/NeuralHaircut || exit 1

# Run rendering for training views
echo -e "\e[1;31;43mRunning rendering for training views\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python render_gaussians.py \
    -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" \
    --skip_test --scene_suffix "_cropped" --iteration 30000 \
    --trainable_cameras --trainable_intrinsics --use_barf || exit 1

# Get FLAME mesh scalp maps
echo -e "\e[1;31;43mGetting FLAME mesh scalp maps\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python extract_non_visible_head_scalp.py \
    --project_dir $PROJECT_DIR/ext/NeuralHaircut --data_dir $DATA_PATH \
    --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
    --cams_path $DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1/cameras/30000_matrices.pkl \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" || exit 1

# Run latent hair strands reconstruction
echo -e "\e[1;31;43mRunning latent hair strands reconstruction\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python train_latent_strands.py \
    -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
    --model_path_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2" \
    --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
    --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
    --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
    --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
    --trainable_cameras --trainable_intrinsics --use_barf \
    --iterations 20000 --port "800$GPU" || exit 1

# Run hair strands reconstruction
echo -e "\e[1;31;43mRunning hair strands reconstruction\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python train_strands.py \
    -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
    --model_path_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
    --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
    --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
    --start_checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
    --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
    --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
    --position_lr_init 0.0000016 --position_lr_max_steps 10000 \
    --trainable_cameras --trainable_intrinsics --use_barf \
    --iterations 10000 --port "800$GPU" || exit 1

rm -rf "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1/train_cropped"

##################
# VISUALIZATIONS #
##################

# Export the resulting strands as pkl and ply
echo -e "\e[1;31;43mExporting the resulting strands as pkl and ply\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python export_curves.py \
    --data_dir $DATA_PATH --model_name $EXP_NAME_3 --iter 10000 \
    --flame_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/stage_3/mesh_final.obj" \
    --scalp_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/scalp_data/scalp.obj" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" || exit 1

# Render the visualizations
echo -e "\e[1;31;43mRendering the visualizations\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
CUDA_VISIBLE_DEVICES="$GPU" python render_video.py \
    --blender_path "$BLENDER_DIR" --input_path "$DATA_PATH" \
    --exp_name_1 "$EXP_NAME_1" --exp_name_3 "$EXP_NAME_3" || exit 1

# Render the strands
echo -e "\e[1;31;43mRendering the strands\e[0m"
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python render_strands.py \
    -s $DATA_PATH --data_dir "$DATA_PATH" --data_device 'cpu' --skip_test \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iteration 30000 \
    --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
    --model_hair_path "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
    --checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
    --checkpoint_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3/checkpoints/10000.pth" \
    --pointcloud_path_head "$EXP_PATH_1/point_cloud/iteration_30000/raw_point_cloud.ply" \
    --interpolate_cameras || exit 1

# # Make the video
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python concat_video.py \
#     --input_path "$DATA_PATH" --exp_name_3 "$EXP_NAME_3"
