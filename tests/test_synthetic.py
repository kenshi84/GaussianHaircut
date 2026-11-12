"""Regression checks for fixed COLMAP calibration and non-destructive case staging.

Run with the gaussian_splatting_hair Python environment from the repository root.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'ext/NeuralHaircut'),
               str(PROJECT / 'ext/NeuralHaircut/k-diffusion')]
from arguments import ModelParams, get_combined_args
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat, Camera as ColmapCamera
from scene.dataset_readers import CameraInfo, readColmapCameras
from utils.camera_utils import loadCam, camera_to_JSON
from utils.graphics_utils import clamp_camera_rays, focal2fov

spec = importlib.util.spec_from_file_location('prepare_synthetic', PROJECT / 'src/preprocessing/preprocess_synthetic_scene.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

CASES = PROJECT / 'datasets/synth/ct2hair'
CASE = CASES / 'bangs'
ARTIFACTS = PROJECT / 'output/synth/ct2hair/bangs/validation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CASE.is_dir():
            raise unittest.SkipTest('ct2hair test cases are not present')
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.cameras = read_intrinsics_binary(CASE / 'sparse/cameras.bin')
        cls.images = read_extrinsics_binary(CASE / 'sparse/images.bin')

    def test_all_cases_and_output_paths(self):
        cases = sorted(p for p in CASES.iterdir() if p.is_dir())
        self.assertEqual(len(cases), 10)
        for case in cases:
            _, names = prepare.validate_colmap_case(case)
            self.assertEqual(len(names), 80)
            self.assertEqual(prepare.default_output_path(case), PROJECT / 'output/synth/ct2hair' / case.name)

    def test_staging_preserves_source_and_mesh(self):
        files = [CASE / 'head_flame.obj', *sorted((CASE / 'sparse').glob('*.bin')),
                 *sorted((CASE / 'cropped/images').glob('*.jpg'))]
        before = {p: (p.stat().st_mtime_ns, digest(p)) for p in files}
        source_tree = set(CASE.rglob('*'))
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as work:
            args = Namespace(data_path=str(CASE), output_path=work, check_only=False, experiment='stage1')
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare_colmap_case(args)
            out = Path(work)
            self.assertEqual(digest(out / 'head_flame.obj'), before[CASE / 'head_flame.obj'][1])
            self.assertEqual(digest(out / 'flame_fitting/stage1/stage_3/mesh_final.obj'), before[CASE / 'head_flame.obj'][1])
            for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
                self.assertEqual(digest(out / 'sparse/0' / name), before[CASE / 'sparse' / name][1])
            self.assertEqual(len(list((out / 'images').glob('*.jpg'))), 80)
            args.output_path = str(CASE / 'must_not_be_written')
            with self.assertRaises(ValueError):
                prepare.prepare_colmap_case(args)
        self.assertEqual(source_tree, set(CASE.rglob('*')))
        self.assertEqual(before, {p: (p.stat().st_mtime_ns, digest(p)) for p in files})

    def test_fixed_flag_and_saved_config(self):
        parser = ArgumentParser()
        model = ModelParams(parser)
        defaults = model.extract(parser.parse_args([]))
        self.assertTrue(defaults.trainable_cameras)
        fixed = model.extract(parser.parse_args(['--fixed_cameras', '--trainable_cameras', '--trainable_intrinsics']))
        self.assertFalse(fixed.trainable_cameras)
        self.assertFalse(fixed.trainable_intrinsics)
        self.assertFalse(fixed.use_barf)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as work:
            (Path(work) / 'cfg_args').write_text(str(Namespace(**vars(fixed))))
            parser = ArgumentParser()
            model = ModelParams(parser, sentinel=True)
            with patch.object(sys, 'argv', ['render', '-m', work]), contextlib.redirect_stdout(io.StringIO()):
                restored = model.extract(get_combined_args(parser))
            self.assertTrue(restored.fixed_cameras)
            self.assertFalse(restored.trainable_cameras)
            self.assertFalse(restored.trainable_intrinsics)

    def test_simple_pinhole_and_camera_export(self):
        im = next(iter(self.images.values()))
        original = self.cameras[im.camera_id]
        simple = ColmapCamera(original.id, 'SIMPLE_PINHOLE', original.width, original.height,
                              np.array([9000., 500., 3000.]))
        with contextlib.redirect_stdout(io.StringIO()):
            info, = readColmapCameras({im.id: im}, {simple.id: simple}, CASE / 'cropped/images')
        exported = camera_to_JSON(0, info)
        self.assertEqual((exported['cx'], exported['cy']), (500., 3000.))
        self.assertAlmostEqual(exported['fx'], 9000.)
        self.assertAlmostEqual(exported['fy'], 9000.)
        json.dumps(exported)
        info.image.close()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required by the renderer cameras')
    def test_projection_auxiliary_paths_and_frozen_cameras(self):
        vertices = np.array([[float(x) for x in line.split()[1:4]]
                             for line in (CASE / 'head_flame.obj').read_text().splitlines()
                             if line.startswith('v ')])
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as work:
            work = Path(work)
            for directory in ('images_2', 'masks_2/body', 'masks_2/hair', 'orientations_2/angles', 'orientations_2/vars'):
                (work / directory).mkdir(parents=True)
            # Includes ordinary, strongly off-axis, and outside-image principal points.
            outside = next(im for im in self.images.values()
                           if self.cameras[im.camera_id].params[3] > self.cameras[im.camera_id].height)
            for im in (self.images[0], self.images[30], outside):
                calibration = self.cameras[im.camera_id]
                fx, fy, cx, cy = calibration.params
                for resolution in (1, 2):
                    size = (calibration.width // 2, calibration.height // 2)
                    image_path = work / 'images_2' / im.name
                    Image.new('RGB', size, 'white').save(image_path)
                    for directory in ('masks_2/body', 'masks_2/hair', 'orientations_2/angles'):
                        Image.new('L', size, 90).save(work / directory / f'{Path(im.name).stem}.png')
                    np.save(work / 'orientations_2/vars' / f'{Path(im.name).stem}.npy', np.ones(size[::-1], dtype=np.float32))
                    info = CameraInfo(im.id, qvec2rotmat(im.qvec).T, im.tvec,
                                      focal2fov(float(fy), calibration.height), focal2fov(float(fx), calibration.width),
                                      Image.open(image_path), str(image_path), Path(im.name).stem,
                                      calibration.width, calibration.height, float(cx), float(cy))
                    args = SimpleNamespace(resolution=resolution, binarize_masks=False, data_device='cpu',
                                           white_background=False, trainable_cameras=False, trainable_intrinsics=False,
                                           use_barf=False, zfar=100000.)
                    camera = loadCam(args, 0, info, 1.)
                    info.image.close()
                    self.assertEqual(list(camera.parameters()), [])
                    initial = camera.full_proj_transform.clone()
                    xyz = torch.tensor(vertices, dtype=torch.float32, device='cuda')
                    hom = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1) @ initial
                    ndc = hom[:, :2] / hom[:, 3:]
                    pixels = ((ndc + 1) * torch.tensor([camera.image_width, camera.image_height], device='cuda') / 2).cpu().numpy()
                    view = vertices @ qvec2rotmat(im.qvec).T + im.tvec
                    expected = (view[:, :2] / view[:, 2:] * [fx, fy] + [cx, cy])
                    expected *= [camera.image_width / calibration.width, camera.image_height / calibration.height]
                    np.testing.assert_allclose(pixels, expected, atol=0.002, rtol=0)
                    visible = (expected[:, 0] >= 0) & (expected[:, 0] < camera.image_width) & (expected[:, 1] >= 0) & (expected[:, 1] < camera.image_height)
                    tx, ty, tz = torch.tensor(view[visible], dtype=torch.float32, device='cuda').T
                    clamped = clamp_camera_rays(tx, ty, tz, camera, torch.tan(camera.FoVx / 2), torch.tan(camera.FoVy / 2))
                    torch.testing.assert_close(clamped[0], tx)
                    torch.testing.assert_close(clamped[1], ty)
                    point = torch.nn.Parameter(xyz[:1].clone())
                    optimizer = torch.optim.Adam([point], lr=0.1)
                    for _ in range(2):
                        optimizer.zero_grad()
                        loss = ((torch.cat((point, torch.ones_like(point[:, :1])), 1) @ camera.full_proj_transform)[:, :2] ** 2).sum()
                        loss.backward()
                        optimizer.step()
                    self.assertTrue(torch.equal(initial, camera.full_proj_transform))
                    del camera

    def test_centered_ray_bounds(self):
        camera = SimpleNamespace(principal_offset_x=0., principal_offset_y=0.)
        tx = torch.tensor([-4., -0.5, 0., 0.5, 4.])
        tz = torch.ones_like(tx)
        x, y = clamp_camera_rays(tx, tx, tz, camera, 0.5, 0.75)
        torch.testing.assert_close(x, tx.clamp(-0.65, 0.65))
        torch.testing.assert_close(y, tx.clamp(-0.975, 0.975))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required by orientation filtering')
    def test_orientation_generation_from_jpg_and_png_mask(self):
        spec = importlib.util.spec_from_file_location('orientation', PROJECT / 'src/preprocessing/calc_orientation_maps.py')
        orientation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orientation)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as work:
            work = Path(work)
            (work / 'images').mkdir()
            (work / 'masks').mkdir()
            pixels = np.random.default_rng(0).integers(0, 256, (12, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(work / 'images/0.jpg')
            Image.new('L', (16, 12), 255).save(work / 'masks/0.png')
            args = Namespace(img_path=str(work / 'images'), mask_path=str(work / 'masks'),
                             orient_dir=str(work / 'angles'), conf_dir=str(work / 'vars'),
                             filtered_img_dir=str(work / 'filtered'), vis_img_dir=str(work / 'vis'),
                             dog_low=0.4, dog_high=10., num_frequencies=1, num_filters=180,
                             num_sigmas_x=1, num_sigmas_y=1, num_offsets=1, patch_size=64)
            orientation.main(args)
            with Image.open(work / 'angles/0.png') as angles:
                self.assertEqual(angles.size, (16, 12))
            variance = np.load(work / 'vars/0.npy')
            self.assertEqual(variance.shape, (12, 16))
            self.assertTrue(np.isfinite(variance).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
