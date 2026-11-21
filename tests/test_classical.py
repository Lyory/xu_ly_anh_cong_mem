import cv2
import numpy as np
import pytest
from src.classical.edge_detection import METHODS as EDGE_METHODS, detect_edges
from src.classical.denoise import METHODS as FILTER_METHODS, denoise
from src.classical.deskew import rotate, estimate_skew, deskew
from src.classical.detect_document import detect_document, rectify_document
from src.classical.deblur import deblur
from src.classical.frequency import motion_psf, fft_spectrum, estimate_motion_psf
from src.data.degradations import add_warp, add_shadow, add_blur, add_noise_jpeg
from src.utils import identity_grid, sample_grid
from src.metrics import image_metrics, local_distortion
from src.pipeline import DocRestorer


@pytest.fixture
def page():
    image = np.full((320, 256, 3), 255, np.uint8)
    for y in range(40, 290, 25):
        cv2.putText(image, 'Document text 123', (12, y), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def test_backward_map_restores_coordinates():
    # Smooth coordinate ramp avoids conflating resampling blur with map direction.
    grid = identity_grid(128, 160)
    image = np.stack([(grid[..., 0]+1)*100, (grid[..., 1]+1)*100, np.full((128, 160), 100)], -1).astype(np.uint8)
    warped, bmap = add_warp(image, np.random.default_rng(12))
    recovered = sample_grid(warped, bmap)
    error = np.abs(recovered[8:-8, 8:-8].astype(float)-image[8:-8, 8:-8]).mean()
    assert error < 1.2
    assert np.abs(warped.astype(float)-image).mean() > error * 2


def test_degradation_reproducibility(page):
    for function in (add_shadow, add_blur, add_noise_jpeg):
        assert np.array_equal(function(page, np.random.default_rng(42)), function(page, np.random.default_rng(42)))
    warped, bmap = add_warp(page, np.random.default_rng(42), strength=0, perspective=False)
    assert np.array_equal(warped, page)
    np.testing.assert_allclose(bmap, identity_grid(*page.shape[:2]), atol=1e-6)


@pytest.mark.parametrize('method', EDGE_METHODS)
def test_edge_operators(page, method):
    edges = detect_edges(page, method)
    assert edges.dtype == np.uint8 and edges.shape == page.shape[:2]
    assert np.any(edges) and not np.all(edges)


@pytest.mark.parametrize('method', FILTER_METHODS)
def test_denoisers_constant_image(method):
    image = np.full((32, 48, 3), 128, np.uint8)
    output = denoise(image, method)
    assert output.shape == image.shape and np.isfinite(output).all()
    assert np.abs(output.astype(float)-image).max() <= 1


def test_median_salt_pepper(page):
    rng = np.random.default_rng(1)
    mask = rng.random(page.shape[:2])
    noisy = page.copy()
    noisy[mask < .03] = 0
    noisy[mask > .97] = 255
    # Use flat paper regions: median may erase tiny strokes, so evaluate background.
    flat = page[..., 0] == 255
    restored = denoise(noisy, 'median')
    assert np.mean((restored[flat].astype(float)-255)**2) < np.mean((noisy[flat].astype(float)-255)**2)


@pytest.mark.parametrize('method', ['hough', 'projection', 'min_area_rect'])
def test_deskew_sign(page, method):
    tilted = rotate(page, 5)
    correction = estimate_skew(tilted, method)
    assert abs(correction+5) < 2, (method, correction)
    restored, _ = deskew(tilted, method)
    assert abs(estimate_skew(restored, method)) < 2


def test_document_detection_and_rectification():
    image = np.full((360, 320, 3), 45, np.uint8)
    corners = np.int32([[35, 25], [290, 45], [280, 330], [20, 320]])
    cv2.fillConvexPoly(image, corners, (245, 245, 245))
    detected = detect_document(image)
    assert detected is not None
    assert np.linalg.norm(detected-corners, axis=1).mean() < 5
    output, matrix = rectify_document(image, detected)
    assert output.mean() > 225 and matrix.shape == (3, 3)
    assert detect_document(np.full_like(image, 255)) is None


@pytest.mark.parametrize('method', ['unsharp', 'wiener', 'richardson_lucy'])
def test_deconvolution_finite(page, method):
    output = deblur(page, method)
    assert output.shape == page.shape and output.dtype == np.uint8


def test_metrics_identity_and_map_units(page):
    metrics = image_metrics(page, page)
    assert metrics['psnr'] == float('inf')
    assert metrics['ssim'] == pytest.approx(1)
    assert metrics['ms_ssim'] == pytest.approx(1)
    base = identity_grid(*page.shape[:2])
    shifted = base.copy()
    shifted[..., 0] += 2/(page.shape[1]-1)
    assert local_distortion(shifted, base) == pytest.approx(1, abs=1e-4)
    with pytest.raises(ValueError):
        image_metrics(page, page[:100])


def test_pipeline_stages_and_missing_dl(page):
    result = DocRestorer(modules={'perspective': False, 'deskew': False}, max_size=None).restore(page)
    stages = list(result['intermediates'])
    assert stages.index('denoise') < stages.index('deblur')
    assert result['image'].shape == page.shape
    assert 'scores' in result['metadata']
    with pytest.raises(ValueError, match='requires'):
        DocRestorer(mode='dl')


def test_frequency_finite(page):
    assert np.isfinite(fft_spectrum(page)).all()
    assert motion_psf(7, 30).sum() == pytest.approx(1)
    psf, metadata = estimate_motion_psf(page)
    assert psf.sum() == pytest.approx(1) and metadata['heuristic']


def test_deblur_only_backend_keeps_classical_shadow_removal(page):
    class DeblurBackend:
        tasks = ('deblur',)

        def __call__(self, image):
            return image.copy()

    shadowed = add_shadow(page, np.random.default_rng(5))
    restorer = DocRestorer('hybrid', restore_backend=DeblurBackend(),
                          modules={'perspective': False, 'deskew': False, 'dewarp': False, 'denoise': False})
    result = restorer.restore(shadowed)
    assert 'shadow' in result['intermediates'] and 'restore_dl' in result['intermediates']
    assert 'deblur' not in result['intermediates']
    assert result['image'].mean() > shadowed.mean()
