"""Global chroma residual energy: candidate for diffuse low-amplitude triggers."""
import numpy as np


def global_chroma_scores(images: np.ndarray, batch_size: int = 128) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 4:
        raise ValueError('Expected NHWC or NCHW batch')
    if x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = x.transpose(0, 2, 3, 1)
    if x.shape[-1] < 3 or min(x.shape[1:3]) < 4:
        raise ValueError('Expected RGB image >=4x4')
    out = np.empty(len(x), np.float64)
    for i in range(0, len(x), batch_size):
        z = x[i:i+batch_size, ..., :3].astype(np.float32)
        if x.dtype.kind in 'ui':
            z /= 255
        elif not (np.isfinite(z).all() and z.min() >= 0 and z.max() <= 1):
            raise ValueError('Expected float image in [0,1]')
        r, g, b = z[..., 0], z[..., 1], z[..., 2]
        channels = (r-b, g-.5*(r+b))
        score = np.zeros(len(z), np.float64)
        for ch in channels:
            hp = ch[:,1:-1,1:-1]-.25*(ch[:,:-2,1:-1]+ch[:,2:,1:-1]+ch[:,1:-1,:-2]+ch[:,1:-1,2:])
            score += (hp*hp).mean(axis=(1,2))
        out[i:i+len(z)] = score
    return out
