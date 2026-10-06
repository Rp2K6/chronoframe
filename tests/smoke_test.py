import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from temporal_blocks import TemporalBlock
from interpolation import JointInterpolator, SingleFrameInterpolator
from temporal_decoder import TinyDecoder

B, T = 2, 6
# 1) Temporal block is identity at init (so pretrained spatial model is preserved)
x = torch.randn(B * T, 64, 8, 8)
blk = TemporalBlock(64)
assert torch.allclose(blk(x, T), x), "TemporalBlock must start as identity"
blk.alpha.data.fill_(1.0)
assert blk(x, T).shape == x.shape

# 2) Interpolation: T keyframes -> 4T-3 frames
k = torch.randn(B, T, 4, 8, 8)
for cls in (JointInterpolator, SingleFrameInterpolator):
    m = cls().eval()
    with torch.no_grad():
        t0 = time.time(); y = m(k); dt = time.time() - t0
    assert y.shape == (B, 4 * T - 3, 4, 8, 8), y.shape
    print(f"{cls.__name__:24s} out={tuple(y.shape)} time={dt*1000:.1f} ms")

# 3) Decoder shapes
z = torch.randn(B, T, 4, 8, 8)
for flag in (False, True):
    assert TinyDecoder(temporal=flag)(z).shape == (B, T, 3, 32, 32)
print("All smoke tests passed.")
