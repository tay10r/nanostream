#!/usr/bin/env python3

# This script iterates the training dataset and builds a covariance
# matrix. From that matrix, the eigen vectors and values are estimated.
# The top K eigen vectors are exported to C, where they are used to
# generate eigen values for a given 3x8x8 block. Each image in the
# training dataset is sampled N times for 3x8x8 blocks.

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

D: int = 3 * 8 * 8

def list_pngs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.png')

def load_rgb_float01(path: Path) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0

def sample_blocks(img: np.ndarray, n_samples: int, block: int = 8) -> np.ndarray:
    h, w, c = img.shape
    if c != 3 or h < block or w < block or n_samples <= 0:
        return np.empty((0, D), dtype=np.float64)

    ys = np.random.randint(0, h - block + 1, size=n_samples, dtype=np.int64)
    xs = np.random.randint(0, w - block + 1, size=n_samples, dtype=np.int64)

    out = np.empty((n_samples, D), dtype=np.float64)
    for i in range(n_samples):
        y = int(ys[i])
        x = int(xs[i])
        patch = img[y:y + block, x:x + block, :]
        vec = patch.transpose(2, 0, 1).reshape(-1)
        out[i, :] = vec.astype(np.float64, copy=False)
    return out

def fit_pca_batch(train_dir: Path, n_per_image: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    paths = list_pngs(train_dir)
    if not paths:
        raise FileNotFoundError(f'No PNGs found in {train_dir}')

    blocks_list: list[np.ndarray] = []
    for p in paths:
        img = load_rgb_float01(p)
        blocks = sample_blocks(img, n_per_image, block=8)
        if blocks.shape[0] > 0:
            blocks_list.append(blocks)

    if not blocks_list:
        raise RuntimeError('No blocks were sampled. Check image sizes and n_per_image.')

    X = np.concatenate(blocks_list, axis=0)
    if X.shape[0] < 2:
        raise RuntimeError('Need at least 2 total samples to compute covariance.')

    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    return mu.astype(np.float32, copy=False), eigvecs.astype(np.float32, copy=False)

def format_c_float(x: float) -> str:
    s = f'{x:.9e}'
    return s + 'f'

def write_pca_c_header(path: Path, eigvecs: np.ndarray, mean: np.ndarray, k: int) -> None:
    if mean.shape != (D,):
        raise ValueError(f'mean must have shape ({D},), got {mean.shape}')
    if eigvecs.shape != (D, D):
        raise ValueError(f'eigvecs must have shape ({D}, {D}), got {eigvecs.shape}')
    if k < 1 or k > D:
        raise ValueError(f'k must be in [1, {D}], got {k}')

    Vk = eigvecs[:, :k].T.astype(np.float32, copy=False)
    mean_f = mean.astype(np.float32, copy=False)

    lines: list[str] = []
    lines.append('#include "nanostream.h"')
    lines.append('')
    lines.append('#include <stdint.h>')
    lines.append('')

    lines.append(f'const float nanostream_mean[{D}] = {{')
    mean_vals = ', '.join(format_c_float(float(mean_f[i])) for i in range(D))
    lines.append(f'  {mean_vals}')
    lines.append('};')
    lines.append('')

    lines.append(f'const float nanostream_eigen_values[{k}][{D}] = {{')
    for r in range(k):
        row_vals = ', '.join(format_c_float(float(Vk[r, c])) for c in range(D))
        lines.append(f'  {{ {row_vals} }},')
    lines.append('};')
    lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', type=Path, default=Path('data'))
    ap.add_argument('--n_per_image', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--output', type=Path, default=Path('nanostream_eigen.c'))
    args = ap.parse_args()

    mean, eigvecs = fit_pca_batch(args.train_dir, args.n_per_image, args.seed)
    write_pca_c_header(args.output, eigvecs, mean, args.k)

if __name__ == '__main__':
    main()