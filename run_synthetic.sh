#!/usr/bin/env bash
# Conda activation scripts can reference unset toolchain variables.
set -eo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_PATH:?Set DATA_PATH to a case containing cropped/images, sparse, and head_flame.obj}"
GPU="${GPU:-0}"
ITERATIONS_STAGE1="${ITERATIONS_STAGE1:-30000}"
ITERATIONS_STAGE2="${ITERATIONS_STAGE2:-20000}"
ITERATIONS_STAGE3="${ITERATIONS_STAGE3:-10000}"
RESOLUTION="${RESOLUTION:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"

eval "$(conda shell.bash hook)"
conda activate gaussian_splatting_hair
PREPARE="$PROJECT_DIR/src/preprocessing/preprocess_synthetic_scene.py"
DATA_PATH="$(realpath -- "$DATA_PATH")"
OUTPUT_PATH="$(python "$PREPARE" --data_path "$DATA_PATH" --print_output_path)"
python "$PREPARE" --data_path "$DATA_PATH" --check_only
for dependency in NeuralHaircut Matte-Anything; do
    if [[ ! -d "$PROJECT_DIR/ext/$dependency" ]]; then
        printf 'Missing dependency: %s/ext/%s. See install.sh.\n' "$PROJECT_DIR" "$dependency" >&2
        exit 1
    fi
done
mkdir -p "$OUTPUT_PATH/logs"
exec > >(tee -a "$OUTPUT_PATH/logs/pipeline.log") 2>&1
python "$PREPARE" --data_path "$DATA_PATH" --output_path "$OUTPUT_PATH"

cd "$PROJECT_DIR/src/preprocessing"
conda activate matte_anything
python calc_masks.py --data_path "$OUTPUT_PATH" --image_format jpg --max_size 2048
conda activate gaussian_splatting_hair
python resize_images.py --data_path "$OUTPUT_PATH" --keep_all
python calc_orientation_maps.py \
    --img_path "$OUTPUT_PATH/images_2" --mask_path "$OUTPUT_PATH/masks_2/hair" \
    --orient_dir "$OUTPUT_PATH/orientations_2/angles" --conf_dir "$OUTPUT_PATH/orientations_2/vars" \
    --filtered_img_dir "$OUTPUT_PATH/orientations_2/filtered_imgs" --vis_img_dir "$OUTPUT_PATH/orientations_2/vis_imgs"

STAGE1="$OUTPUT_PATH/3d_gaussian_splatting/stage1"
STAGE2="$OUTPUT_PATH/strands_reconstruction/stage2"
STAGE3="$OUTPUT_PATH/curves_reconstruction/stage3"
FLAME="$OUTPUT_PATH/flame_fitting/stage1"
HAIR_CONFIG="$PROJECT_DIR/src/arguments/hair_strands_textured.yaml"
COMMON=(-s "$OUTPUT_PATH" -m "$STAGE1" -r "$RESOLUTION" --fixed_cameras --zfar 100000 --data_device cpu)

cd "$PROJECT_DIR/src"
python train_gaussians.py "${COMMON[@]}" --lambda_dorient 0.1 \
    --iterations "$ITERATIONS_STAGE1" --save_iterations "$ITERATIONS_STAGE1" \
    --checkpoint_iterations "$ITERATIONS_STAGE1" --test_iterations "$ITERATIONS_STAGE1"

cd "$PROJECT_DIR/src/preprocessing"
python scale_scene_into_sphere.py --path_to_data "$OUTPUT_PATH" -m "$STAGE1" --iter "$ITERATIONS_STAGE1"
python filter_flame_intersections.py --flame_mesh_dir "$FLAME" -m "$STAGE1" \
    --iter "$ITERATIONS_STAGE1" --project_dir "$PROJECT_DIR/ext/NeuralHaircut"

cd "$PROJECT_DIR/src"
python render_gaussians.py "${COMMON[@]}" --skip_test --scene_suffix _cropped --iteration "$ITERATIONS_STAGE1"
cd "$PROJECT_DIR/src/preprocessing"
python extract_non_visible_head_scalp.py --data_dir "$OUTPUT_PATH" -m "$STAGE1" \
    --flame_mesh_dir "$FLAME" --fixed_cameras --project_dir "$PROJECT_DIR/ext/NeuralHaircut"

cd "$PROJECT_DIR/src"
HAIR=("${COMMON[@]}" --flame_mesh_dir "$FLAME" --hair_conf_path "$HAIR_CONFIG"
    --pointcloud_path_head "$STAGE1/point_cloud_filtered/iteration_$ITERATIONS_STAGE1/raw_point_cloud.ply"
    --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01
    --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data "$ITERATIONS_STAGE1")
python train_latent_strands.py "${HAIR[@]}" --model_path_hair "$STAGE2" \
    --iterations "$ITERATIONS_STAGE2" --save_iterations "$ITERATIONS_STAGE2" \
    --checkpoint_iterations "$ITERATIONS_STAGE2" --test_iterations "$ITERATIONS_STAGE2"
python train_strands.py "${HAIR[@]}" --model_path_curves "$STAGE3" \
    --start_checkpoint_hair "$STAGE2/checkpoints/$ITERATIONS_STAGE2.pth" \
    --position_lr_init 0.0000016 --position_lr_max_steps "$ITERATIONS_STAGE3" \
    --iterations "$ITERATIONS_STAGE3" --save_iterations "$ITERATIONS_STAGE3" \
    --checkpoint_iterations "$ITERATIONS_STAGE3" --test_iterations "$ITERATIONS_STAGE3"

cd "$PROJECT_DIR/src/preprocessing"
python export_curves.py --data_dir "$OUTPUT_PATH" --model_name stage3 --iter "$ITERATIONS_STAGE3" \
    --flame_mesh_path "$FLAME/stage_3/mesh_final.obj" --scalp_mesh_path "$FLAME/scalp_data/scalp.obj"
printf 'Synthetic reconstruction saved to %s\n' "$OUTPUT_PATH"
