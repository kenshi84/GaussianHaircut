# Gaussian Haircut: Human Hair Reconstruction with Strand-Aligned 3D Gaussians

[**Paper**](https://arxiv.org/abs/2409.14778) | [**Project Page**](https://eth-ait.github.io/GaussianHaircut/)

This repository contains an official implementation of Gaussian Haircut, a strand-based hair reconstruction approach for monocular videos.

## Getting started

1. **Install CUDA 11.8**

   Follow the instructions on https://developer.nvidia.com/cuda-11-8-0-download-archive.

   Make sure that
   - PATH includes <CUDA_DIR>/bin
   - LD_LIBRARY_PATH includes <CUDA_DIR>/lib64

   The environment was tested only with this CUDA version.

2. **Install Blender 3.6** in order to create strand visualizations

   Follow instructions on https://www.blender.org/download/lts/3-6.

3. **Close the repo and run the install script**

   ```bash
   git clone git@github.com:eth-ait/GaussianHaircut.git
   cd GaussianHaircut
   chmod +x ./install.sh
   ./install.sh
   ```

## Reconstruction

1. **Record a monocular video**

   Use examples on the project page as references and introduce as little motion blur as possible.

2. **Setup a directory for the reconstructed scene**

   Put the video file in it and rename it to raw.mp4

3. **Run the script**

   ```bash
   export PROJECT_DIR="[/path/to/]GaussianHaircut"
   export BLENDER_DIR="[/path/to/blender/folder/]blender"
   DATA_PATH="[path/to/scene/folder]" ./run.sh
   ```

   The script performs data pre-processing, reconstruction, and final visualization using Blender. Use Tensorboard to see intermediate visualizations.

## Synthetic cases with known cameras and FLAME meshes

After installing the reconstruction and Matte-Anything dependencies, run:

```bash
DATA_PATH=datasets/synth/ct2hair/bangs ./run_synthetic.sh
```

The input case must contain `cropped/images/*.jpg`, the COLMAP binaries
`sparse/{cameras,images,points3D}.bin`, and `head_flame.obj` in the original
5023-vertex FLAME topology. Cameras must be PINHOLE or SIMPLE_PINHOLE, with
calibration dimensions matching the supplied images. `points3D.bin` supplies
the initial Gaussian point cloud and must be nonempty.

Outputs mirror the path below `datasets/`: this example writes everything to
`output/synth/ct2hair/bangs/`. A case outside `datasets/` defaults to
`output/<case_name>/`. Input files are never moved or edited. The output contains
verbatim camera binaries in `sparse/0/` and a verbatim `head_flame.obj`.

The synthetic runner generates masks from RGB, keeps every registered view,
and skips camera reconstruction, camera optimization, and FLAME fitting.
Half-resolution working images use the corresponding scaled intrinsics,
including each camera's principal point. Gaussian points, scalp roots, and
exported strands retain millimeters and the supplied +Y up, +Z front, +X left
coordinate frame; no alignment or unit conversion is required for GT comparison.
Scene cropping only removes points. `scale.pickle` records the crop radius and
center for the existing strand-thickness calculation; it does not transform the
saved geometry.

Final strands are in
`curves_reconstruction/stage3/strands/10000_strands.pkl` (an `N x 100 x 3` array)
and `10000_strands.ply` (the same samples flattened into a point cloud).
The stage directories also contain checkpoints and camera metadata; the
pipeline log is `logs/pipeline.log`. Blender video generation and GT metrics
are not part of this runner.

The defaults are 30,000 Gaussian, 20,000 latent-strand, and 10,000 refinement
iterations. Override them for a smoke test; its geometry is not a converged
accuracy result:

```bash
DATA_PATH=datasets/synth/ct2hair/bangs \
ITERATIONS_STAGE1=1000 ITERATIONS_STAGE2=10 ITERATIONS_STAGE3=10 \
RESOLUTION=2 ./run_synthetic.sh
```

`RESOLUTION=2` halves the working images again during training/rendering;
the default is `1`. `GPU` selects the GPU (default `0`). All stages use fixed
cameras, a millimeter-appropriate far plane, and CPU storage for training images.
Run cases sequentially:

```bash
for case in datasets/synth/ct2hair/*/; do
    DATA_PATH="$case" ./run_synthetic.sh || break
done
```

Validate input cases without generating reconstruction files using
`python src/preprocessing/preprocess_synthetic_scene.py --data_path <case> --check_only`.
Run regression checks with `python -B tests/test_synthetic.py` in the
`gaussian_splatting_hair` environment. They check all ten local ct2hair cases,
camera projection and freezing, JPG auxiliary-file paths, and input preservation.

## License

This code is based on the 3D Gaussian Splatting project. For terms and conditions, please refer to LICENSE_3DGS. The rest of the code is distributed under CC BY-NC-SA 4.0.

If this code is helpful in your project, cite the papers below.

## Citation

```
@inproceedings{zakharov2024gh,
   title = {Human Hair Reconstruction with Strand-Aligned 3D Gaussians},
   author = {Zakharov, Egor and Sklyarova, Vanessa and Black, Michael J and Nam, Giljoo and Thies, Justus and Hilliges, Otmar},
   booktitle = {European Conference of Computer Vision (ECCV)},
   year = {2024}
} 
```

## Links

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

- [Neural Haircut](https://github.com/SamsungLabs/NeuralHaircut): FLAME fitting pipeline, strand prior and hairstyle diffusion prior

- [HAAR](https://github.com/Vanessik/HAAR): hair upsampling

- [Matte-Anything](https://github.com/hustvl/Matte-Anything): hair and body segmentation

- [PIXIE](https://github.com/yfeng95/PIXIE): initialization for FLAME fitting

- [Face-Alignment](https://github.com/1adrianb/face-alignment), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose): keypoints detection for FLAME fitting
