"""2D U-Net on PS-OCT sections, trained on labels where 255 means "do not learn from this pixel"."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from brainseg import IGNORE
from brainseg.preprocess import normalize, unimaged

N_CLASSES = 4  # background, gray matter, white matter, fluid; damage (4) is not predicted


def device():
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels, width=16, depth=4):
        super().__init__()
        self.config = {'in_channels': in_channels, 'width': width, 'depth': depth}
        c = [width * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(in_channels, c[0])] + [block(c[i], c[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(depth))])
        self.merge = nn.ModuleList([block(2 * c[i], c[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(width, N_CLASSES, 1)

    def forward(self, x):
        skips = []
        for down in self.down[:-1]:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.down[-1](x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips)):
            x = merge(torch.cat([up(x), skip], 1))
        return self.head(x)


def prepare(a):
    """H x W x C section -> C x H x W float32: each contrast scaled to its imaged 1st/99th percentiles, unimaged 0."""
    outside = unimaged(a)
    if outside.all():
        return np.zeros((a.shape[-1],) + a.shape[:2], np.float32)
    x = np.stack([np.nan_to_num(normalize(a[..., c], ~outside)) for c in range(a.shape[-1])])
    x[:, outside] = 0
    return x.astype(np.float32)


def fit(sections, labels, iterations=4000, patch=256, batch=16, lr=1e-3, width=16, depth=4, seed=0, on=None):
    """Train a U-Net on H x W x C sections and their labels (0-3; 255 and damage are ignored); return (model, losses).

    Patches are centred uniformly on eligible labelled pixels and randomly flipped and rotated by multiples of
    90 degrees. Unimaged pixels are never learned from. Per-row cumulative counts keep the sampling index small
    without excluding labels between fixed grid points.
    """
    on = on or device()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xs, ys, row_totals = [], [], []
    half = patch // 2
    for a, y in zip(sections, labels):
        y = np.where((y < N_CLASSES) & ~unimaged(a), y, IGNORE).astype(np.uint8)
        xs.append(np.pad(prepare(a), ((0, 0), (half, half), (half, half))))
        ys.append(np.pad(y, half, constant_values=IGNORE))
        row_totals.append(np.cumsum(np.count_nonzero(ys[-1] != IGNORE, axis=1)))
    keep = [i for i, counts in enumerate(row_totals) if counts[-1]]
    if not keep:
        raise ValueError('No labelled pixels to train on.')
    weights = np.array([row_totals[i][-1] for i in keep], float)
    model = UNet(xs[0].shape[0], width, depth).to(on)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=iterations)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    losses = []
    model.train()
    for _ in range(iterations):
        bx, by = [], []
        for i in rng.choice(keep, batch, p=weights / weights.sum()):
            # Pick a labelled-pixel rank, then locate its row and column without storing every coordinate.
            rank = rng.integers(row_totals[i][-1])
            r = np.searchsorted(row_totals[i], rank, side='right')
            before = row_totals[i][r - 1] if r else 0
            c = np.flatnonzero(ys[i][r] != IGNORE)[rank - before]
            x, y = xs[i][:, r - half : r + half, c - half : c + half], ys[i][r - half : r + half, c - half : c + half]
            k, flip = rng.integers(4), rng.integers(2)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if flip:
                x, y = x[:, :, ::-1], y[:, ::-1]
            bx.append(np.ascontiguousarray(x))
            by.append(np.ascontiguousarray(y))
        x, y = torch.from_numpy(np.stack(bx)).to(on), torch.from_numpy(np.stack(by)).long().to(on)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(loss.item())
    return model.eval().cpu(), losses


@torch.no_grad()
def predict(model, a, on=None):
    """Label a whole H x W x C section with 0-3; unimaged pixels are background."""
    on = on or device()
    model = model.to(on).eval()
    x = prepare(a)
    m = 2 ** model.config['depth']
    h, w = x.shape[1:]
    x = np.pad(x, ((0, 0), (0, -h % m), (0, -w % m)))
    out = model(torch.from_numpy(x)[None].to(on)).argmax(1)[0, :h, :w].cpu().numpy().astype(np.uint8)
    out[unimaged(a)] = 0
    return out


def save(model, path, **metadata):
    torch.save({'config': model.config, 'state': model.state_dict(), 'metadata': metadata}, path)


def load(path):
    saved = torch.load(path, map_location='cpu')  # weights_only by default: no code is executed
    model = UNet(**saved['config'])
    model.load_state_dict(saved['state'])
    return model.eval(), saved['metadata']
