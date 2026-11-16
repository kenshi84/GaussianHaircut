"""Check final strand conversion with the installed hairutil binary."""
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/postprocessing'))
from export_final_strands import export_final_strands


@unittest.skipUnless(shutil.which('hairutil'), 'hairutil is required')
class ExportFinalStrandsTests(unittest.TestCase):
    def test_layout_and_round_trip_on_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / 'case with spaces'
            strands_dir = case / 'work/curves_reconstruction/stage3/strands'
            strands_dir.mkdir(parents=True)
            head = case / 'head_flame.obj'
            head.write_text('# supplied head\n')
            strands = np.arange(600, dtype=np.float32).reshape(2, 100, 3) / 8
            # A second export must replace both final files with the new array.
            for offset in (0, 10):
                expected = strands + offset
                with (strands_dir / '10000_strands.pkl').open('wb') as source:
                    pickle.dump(expected, source)
                export_final_strands(case)
                self.assertEqual({p.name for p in case.iterdir()}, {
                    'work', 'head_flame.obj', '10000_strands.data', '10000_strands.ply',
                })
                self.assertEqual(head.read_text(), '# supplied head\n')
                np.testing.assert_array_equal(np.load(strands_dir / '10000_strands.npy'), expected)
                for extension in ('data', 'ply'):
                    restored = case / 'work' / extension
                    restored.mkdir(exist_ok=True)
                    subprocess.run([
                        'hairutil', 'convert', '-i', str(case / f'10000_strands.{extension}'),
                        '-o', 'npy', '-d', str(restored), '--no-autofix', '--overwrite',
                    ], check=True, capture_output=True)
                    np.testing.assert_array_equal(np.load(restored / '10000_strands.npy'), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
