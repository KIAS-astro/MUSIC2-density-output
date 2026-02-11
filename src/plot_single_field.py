#!/usr/bin/env python3
"""
Plot slices from a single MUSIC2 delta/theta .dat file.

Unlike plot_field_slices.py, this script:
- Plots a single specified .dat file directly
- Does not use grid offsets (plots the data as-is)
- Prints grid dimensions on screen

Usage:
    python plot_single_field.py merged_delta_level8.dat
    python plot_single_field.py delta_level8_real.dat --slice-z 0.5 --slice-y 0.5
    python plot_single_field.py data.dat --output my_plot.png
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse


def read_field(filename):
    """Read a MUSIC2 debug binary field.

    Header: nx(uint64) ny(uint64) nz(uint64) ox(int32) oy(int32) oz(int32)
    Data:   float64[nx*ny*nz] in C order (i,j,k) with k fastest
    """
    with open(filename, 'rb') as f:
        nx = struct.unpack('Q', f.read(8))[0]
        ny = struct.unpack('Q', f.read(8))[0]
        nz = struct.unpack('Q', f.read(8))[0]
        ox = struct.unpack('i', f.read(4))[0]
        oy = struct.unpack('i', f.read(4))[0]
        oz = struct.unpack('i', f.read(4))[0]

        data = np.fromfile(f, dtype=np.float64, count=nx * ny * nz)
        data = data.reshape((int(nx), int(ny), int(nz)), order='C')

    return data, (int(nx), int(ny), int(nz)), (ox, oy, oz)


def main():
    parser = argparse.ArgumentParser(
        description="Plot slices from a single MUSIC2 .dat file.")
    parser.add_argument("file", help="Path to the .dat file")
    parser.add_argument("--slice-z", type=float, default=0.5, dest="slice_z",
                        help="Fractional z-position for x-y slice (0-1, default 0.5)")
    parser.add_argument("--slice-y", type=float, default=0.5, dest="slice_y",
                        help="Fractional y-position for x-z slice (0-1, default 0.5)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output filename (default: {input}_slice.png)")
    args = parser.parse_args()

    # Read data
    print(f"Reading {args.file}...")
    data, size, offset = read_field(args.file)
    nx, ny, nz = size
    ox, oy, oz = offset

    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    print(f"Grid offset: ({ox}, {oy}, {oz})")
    print(f"Data range: [{data.min():.6e}, {data.max():.6e}]")
    print(f"Data mean: {data.mean():.6e}, std: {data.std():.6e}")

    # Compute slice indices
    z_idx = int(args.slice_z * nz)
    z_idx = max(0, min(z_idx, nz - 1))
    y_idx = int(args.slice_y * ny)
    y_idx = max(0, min(y_idx, ny - 1))

    print(f"\nSlicing at z={z_idx}/{nz}, y={y_idx}/{ny}")

    # Extract slices
    slice_xy = data[:, :, z_idx]  # x-y plane at fixed z
    slice_xz = data[:, y_idx, :]  # x-z plane at fixed y

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Symmetric color scale
    vmax = max(abs(data.min()), abs(data.max()))
    cmap = plt.cm.RdBu_r

    # x-y slice (left)
    ax = axes[0]
    im = ax.imshow(slice_xy.T, origin='lower', cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect='equal', interpolation='none')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\delta$')
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    ax.set_title(f'x-y slice @ z={z_idx}/{nz}')

    # x-z slice (right)
    ax = axes[1]
    im = ax.imshow(slice_xz.T, origin='lower', cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect='equal', interpolation='none')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\delta$')
    ax.set_xlabel('x index')
    ax.set_ylabel('z index')
    ax.set_title(f'x-z slice @ y={y_idx}/{ny}')

    fig.suptitle(f'{args.file}\nGrid: {nx} x {ny} x {nz}', fontsize=11)
    fig.tight_layout()

    # Save
    if args.output:
        outfile = args.output
    else:
        base = args.file.rsplit('.', 1)[0]
        outfile = f"{base}_slice.png"

    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"\nSaved {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    main()
