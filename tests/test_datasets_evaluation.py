import json
import cv2
import numpy as np
import pytest
from src.utils import write_rgb
from scripts.dataset_common import generate
from scripts.evaluate import evaluate


def test_split_groups_and_paired_evaluation(tmp_path):
    clean = tmp_path/'clean'
    for document in range(5):
        for page in range(2):
            image = np.full((64, 80, 3), 255, np.uint8)
            cv2.putText(image, f'{document} {page}', (5, 40), 0, .7, (0, 0, 0), 1)
            write_rgb(clean/f'doc{document}__p{page}.png', image)
    root = tmp_path/'synthetic'
    generate('warp', ['--clean', str(clean), '--output', str(root), '--size', '64', '--variants', '1'])
    manifest = json.loads((root/'manifest.json').read_text())
    memberships = {}
    for item in manifest['samples']:
        memberships.setdefault(item['group'], set()).add(item['split'])
    assert all(len(splits) == 1 for splits in memberships.values())
    assert {s for splits in memberships.values() for s in splits} == {'train', 'val', 'test'}
    test = root/'test'
    rows = evaluate(test/'input', tmp_path/'result.csv', [{'name': 'input'}, {'name': 'classical'}], test/'target', test/'bmap')
    assert len(rows) == 4
    assert all(np.isfinite(row['psnr']) for row in rows)
    assert (tmp_path/'result_summary.csv').exists()
    with pytest.raises(ValueError, match='fixed coordinates'):
        evaluate(test/'input', tmp_path/'bad.csv', [{'name': 'bad', 'options': {'modules': {'deskew': True}}}], test/'target')


def test_torch_dataset_maps(tmp_path):
    pytest.importorskip('torch')
    from src.data.datasets import PairedDataset
    from src.utils import identity_grid
    for directory in ['input', 'target']:
        write_rgb(tmp_path/'train'/directory/'sample.png', np.full((64, 80, 3), 255, np.uint8))
    (tmp_path/'train'/'bmap').mkdir()
    np.save(tmp_path/'train'/'bmap'/'sample.npy', identity_grid(64, 80))
    sample = PairedDataset(tmp_path, task='warp', size=64)[0]
    assert sample['input'].shape == (3, 64, 64)
    assert sample['bmap'].shape == (64, 64, 2)
