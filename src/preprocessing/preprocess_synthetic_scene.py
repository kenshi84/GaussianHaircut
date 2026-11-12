import os
from argparse import ArgumentParser
import pickle as pkl
import shutil
import sys
from pathlib import Path


def default_output_path(data_path):
    project = Path(__file__).resolve().parents[2]
    source = Path(data_path).resolve()
    try:
        relative = source.relative_to((project / 'datasets').resolve())
    except ValueError:
        relative = Path(source.name)
    return project / 'output' / relative


def validate_colmap_case(data_path):
    # Import the standalone COLMAP reader without importing the CUDA scene package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scene'))
    from colmap_loader import read_intrinsics_binary, read_extrinsics_binary, read_points3D_binary
    import numpy as np
    from PIL import Image

    source = Path(data_path).resolve()
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        if not (source / 'sparse' / name).is_file():
            raise ValueError(f'Missing COLMAP file: {source / "sparse" / name}')
    cameras = read_intrinsics_binary(source / 'sparse/cameras.bin')
    images = read_extrinsics_binary(source / 'sparse/images.bin')
    if not images:
        raise ValueError('The case has no registered images')
    names = set()
    for image in images.values():
        if Path(image.name).name != image.name or Path(image.name).suffix.lower() != '.jpg':
            raise ValueError(f'Expected a JPG basename in images.bin: {image.name}')
        if image.name in names:
            raise ValueError(f'Duplicate image name: {image.name}')
        names.add(image.name)
        camera = cameras[image.camera_id]
        if camera.model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError(f'Unsupported camera model: {camera.model}; supply undistorted pinhole cameras')
        if not np.isfinite(camera.params).all() or min(camera.params[:2 if camera.model == 'PINHOLE' else 1]) <= 0:
            raise ValueError(f'Invalid intrinsics for {image.name}')
        if not np.isfinite(image.qvec).all() or not np.isfinite(image.tvec).all() or not np.isclose(np.linalg.norm(image.qvec), 1):
            raise ValueError(f'Invalid pose for {image.name}')
        with Image.open(source / 'cropped/images' / image.name) as rgb:
            if rgb.size != (camera.width, camera.height):
                raise ValueError(f'Image/calibration dimensions differ: {image.name}')
    xyz, _, _ = read_points3D_binary(source / 'sparse/points3D.bin')
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError('points3D.bin must contain finite initialization points')
    vertices, faces = [], []
    with (source / 'head_flame.obj').open() as mesh:
        for line in mesh:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) for x in line.split()[1:]])
    if len(vertices) != 5023 or len(faces) != 9976 or any(len(f) != 3 for f in faces):
        raise ValueError('head_flame.obj must use the 5023-vertex, 9976-triangle FLAME topology')
    if not np.isfinite(vertices).all() or min(map(min, faces)) < 1 or max(map(max, faces)) > len(vertices):
        raise ValueError('Invalid FLAME vertices or face indices')
    return source, sorted(names)


def prepare_colmap_case(args):
    source, names = validate_colmap_case(args.data_path)
    if args.check_only:
        print(f'{source.name}: {len(names)} calibrated images; valid COLMAP/FLAME input')
        return
    output = Path(args.output_path).resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the input case directory')
    for directory in ('images', 'sparse/0', f'flame_fitting/{args.experiment}/stage_3'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source / 'cropped/images' / name, output / 'images' / name)
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        shutil.copy2(source / 'sparse' / name, output / 'sparse/0' / name)
    # Copy the OBJ verbatim: mesh loaders must not reindex or normalize the supplied head.
    shutil.copy2(source / 'head_flame.obj', output / 'head_flame.obj')
    shutil.copy2(source / 'head_flame.obj', output / f'flame_fitting/{args.experiment}/stage_3/mesh_final.obj')
    print(f'Prepared {len(names)} views in {output}; coordinates remain in millimeters')



def main(args):
    import torch
    from torchvision.transforms import ToPILImage

    data_path = args.data_path
    os.makedirs(f'{data_path}/images', exist_ok=True)
    os.makedirs(f'{data_path}/masks/hair', exist_ok=True)
    os.makedirs(f'{data_path}/masks/body', exist_ok=True)
    os.makedirs(f'{data_path}/orientations/angles', exist_ok=True)
    os.makedirs(f'{data_path}/orientations/vars', exist_ok=True)
    os.makedirs(f'{data_path}/flame_fitting/scalp_data', exist_ok=True)
    os.makedirs(f'{data_path}/flame_fitting/stage_3', exist_ok=True)
    filenames = os.listdir(f'{data_path}/image')
    basenames = [filename.split('.')[0].split('_')[1] for filename in filenames]
    for basename in basenames:
        shutil.move(f'{data_path}/image/img_{basename}.png', f'{data_path}/images/{basename}.png')
        shutil.move(f'{data_path}/mask/img_{basename}.png', f'{data_path}/masks/body/{basename}.png')
        shutil.move(f'{data_path}/hair_mask/img_{basename}.png', f'{data_path}/masks/hair/{basename}.png')
        shutil.move(f'{data_path}/orientation_maps/img_{basename}.png', f'{data_path}/orientations/angles/{basename}.png')
        shutil.move(f'{data_path}/confidence_maps/img_{basename}.npy', f'{data_path}/orientations/vars/{basename}.npy')
    os.rmdir(f'{data_path}/image')
    os.rmdir(f'{data_path}/mask')
    os.rmdir(f'{data_path}/hair_mask')
    os.rmdir(f'{data_path}/orientation_maps')
    os.rmdir(f'{data_path}/confidence_maps')
    shutil.move(f'{data_path}/cut_scalp_verts.pickle', f'{data_path}/flame_fitting/scalp_data/cut_scalp_verts.pickle')
    dif_mask = 1 - torch.load(f'{data_path}/dif_mask.pth')
    ToPILImage()(dif_mask).save(f'{data_path}/flame_fitting/scalp_data/dif_mask.png')
    os.remove(f'{data_path}/dif_mask.pth')
    shutil.move(f'{data_path}/scaled_head_prior.obj', f'{data_path}/flame_fitting/stage_3/mesh_final.obj')

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--output_path', type=str, help='Stage a COLMAP/FLAME case here without modifying its inputs')
    parser.add_argument('--experiment', default='stage1')
    parser.add_argument('--check_only', action='store_true')
    parser.add_argument('--print_output_path', action='store_true')
    args = parser.parse_args()

    if args.print_output_path:
        print(default_output_path(args.data_path))
    elif args.output_path or args.check_only:
        prepare_colmap_case(args)
    else:
        main(args)
