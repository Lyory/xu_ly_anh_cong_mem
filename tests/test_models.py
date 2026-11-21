import numpy as np
import pytest
torch = pytest.importorskip('torch')
from src.models.unet_attention import UNetAttention
from src.models.dewarp_net import DewarpNet, unwarp, identity_grid
from src.losses import CombinedLoss, DewarpLoss
from src.pipeline import DocRestorer

torch.set_num_threads(2)


@pytest.mark.parametrize('channels', [3, 4])
def test_unet_odd_shapes_and_backward(channels):
    model = UNetAttention(in_ch=channels, base=4)
    image = torch.rand(2, channels, 65, 79)
    target = torch.rand(2, 3, 65, 79)
    prediction = model(image)
    assert prediction.shape == target.shape
    loss = CombinedLoss()(prediction, target)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert torch.isfinite(loss)


def test_grid_identity_and_dewarp_backward():
    image = torch.rand(2, 3, 64, 80)
    zero = torch.zeros(2, 2, 64, 80)
    torch.testing.assert_close(unwarp(image, zero), image, atol=1e-5, rtol=1e-5)
    model = DewarpNet()
    flow = model(image)
    target_map = identity_grid(2, 64, 80).clone()
    target_map[..., 0] += .02
    loss = DewarpLoss()(flow, image, image*.9, target_map)
    loss.backward()
    assert flow.shape == (2, 2, 64, 80) and model.output.weight.grad.abs().sum() > 0


def test_checkpoint_pipeline_roundtrip(tmp_path):
    model = UNetAttention(base=4, in_ch=4)
    checkpoint = tmp_path/'unet.pt'
    torch.save({'model': model.state_dict(), 'task': 'shadow_deblur', 'epoch': 0,
                'config': {'model': {'base': 4, 'in_ch': 4}}, 'smoke_only': True}, checkpoint)
    image = np.full((65, 79, 3), 180, np.uint8)
    restore = DocRestorer('dl', shadow_checkpoint=checkpoint, modules={'perspective': False, 'deskew': False, 'denoise': False})
    result = restore.restore(image)
    np.testing.assert_array_equal(result['image'], image)
    assert result['metadata']['models']['shadow_deblur']['smoke_only']
    assert 'restore_dl' in result['intermediates']
    with pytest.raises(ValueError, match='Joint'):
        DocRestorer('hybrid', shadow_checkpoint=checkpoint, modules={'shadow': True})
