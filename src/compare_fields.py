#!/usr/bin/env python3
"""
Compare two MUSIC2 .dat field files numerically.

Usage:
    python compare_fields.py file1.dat file2.dat
    python compare_fields.py file1.dat file2.dat --plot
"""

import numpy as np
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
        description="Compare two MUSIC2 .dat field files numerically.")
    parser.add_argument("file1", help="First .dat file")
    parser.add_argument("file2", help="Second .dat file")
    parser.add_argument("--plot", action="store_true",
                        help="Plot difference slices")
    parser.add_argument("--output", "-o", default=None,
                        help="Output plot filename (default: compare_diff.png)")
    args = parser.parse_args()

    # Read both files
    print(f"Reading {args.file1}...")
    data1, size1, offset1 = read_field(args.file1)
    print(f"  Size: {size1[0]} x {size1[1]} x {size1[2]}")
    print(f"  Offset: ({offset1[0]}, {offset1[1]}, {offset1[2]})")
    print(f"  Range: [{data1.min():.6e}, {data1.max():.6e}]")
    print(f"  Mean: {data1.mean():.6e}, Std: {data1.std():.6e}")

    print(f"\nReading {args.file2}...")
    data2, size2, offset2 = read_field(args.file2)
    print(f"  Size: {size2[0]} x {size2[1]} x {size2[2]}")
    print(f"  Offset: ({offset2[0]}, {offset2[1]}, {offset2[2]})")
    print(f"  Range: [{data2.min():.6e}, {data2.max():.6e}]")
    print(f"  Mean: {data2.mean():.6e}, Std: {data2.std():.6e}")

    # Check if sizes match
    if size1 != size2:
        print(f"\nWARNING: Grid sizes differ!")
        print(f"  File1: {size1[0]} x {size1[1]} x {size1[2]}")
        print(f"  File2: {size2[0]} x {size2[1]} x {size2[2]}")

        # Try to find overlapping region
        min_nx = min(size1[0], size2[0])
        min_ny = min(size1[1], size2[1])
        min_nz = min(size1[2], size2[2])

        print(f"\nComparing overlapping region: {min_nx} x {min_ny} x {min_nz}")
        data1 = data1[:min_nx, :min_ny, :min_nz]
        data2 = data2[:min_nx, :min_ny, :min_nz]

    # Compute difference
    diff = data1 - data2

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # Basic statistics
    print(f"\nDifference (file1 - file2):")
    print(f"  Max absolute diff:  {np.abs(diff).max():.6e}")
    print(f"  Mean diff:          {diff.mean():.6e}")
    print(f"  RMS diff:           {np.sqrt((diff**2).mean()):.6e}")
    print(f"  Std of diff:        {diff.std():.6e}")

    # Relative difference (avoid division by zero)
    mask = np.abs(data1) > 1e-10
    if mask.any():
        rel_diff = np.abs(diff[mask] / data1[mask])
        print(f"\nRelative difference (where |data1| > 1e-10):")
        print(f"  Max relative diff:  {rel_diff.max():.6e}")
        print(f"  Mean relative diff: {rel_diff.mean():.6e}")

    # Correlation
    corr = np.corrcoef(data1.flatten(), data2.flatten())[0, 1]
    print(f"\nCorrelation coefficient: {corr:.10f}")

    # Check if identical
    if np.allclose(data1, data2, rtol=1e-14, atol=1e-14):
        print("\n*** Fields are IDENTICAL (within machine precision) ***")
    elif np.allclose(data1, data2, rtol=1e-6, atol=1e-10):
        print("\n*** Fields are NEARLY IDENTICAL (rtol=1e-6) ***")
    else:
        print("\n*** Fields DIFFER ***")

    # Find location of maximum difference
    idx = np.unravel_index(np.abs(diff).argmax(), diff.shape)
    print(f"\nMax difference location: ({idx[0]}, {idx[1]}, {idx[2]})")
    print(f"  File1 value: {data1[idx]:.10e}")
    print(f"  File2 value: {data2[idx]:.10e}")
    print(f"  Difference:  {diff[idx]:.10e}")

    # Plot if requested
    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Slice at z=nz/2
        nz = diff.shape[2]
        ny = diff.shape[1]
        z_idx = nz // 2
        y_idx = ny // 2

        vmax1 = max(abs(data1.min()), abs(data1.max()))
        vmax_diff = max(abs(diff.min()), abs(diff.max()))
        if vmax_diff == 0:
            vmax_diff = 1e-10

        cmap = plt.cm.RdBu_r

        # Top row: x-y slices
        ax = axes[0, 0]
        im = ax.imshow(data1[:, :, z_idx].T, origin='lower', cmap=cmap,
                       vmin=-vmax1, vmax=vmax1)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'File1 (z={z_idx})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = axes[0, 1]
        im = ax.imshow(data2[:, :, z_idx].T, origin='lower', cmap=cmap,
                       vmin=-vmax1, vmax=vmax1)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'File2 (z={z_idx})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = axes[0, 2]
        im = ax.imshow(diff[:, :, z_idx].T, origin='lower', cmap=cmap,
                       vmin=-vmax_diff, vmax=vmax_diff)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Difference (z={z_idx})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Bottom row: x-z slices
        ax = axes[1, 0]
        im = ax.imshow(data1[:, y_idx, :].T, origin='lower', cmap=cmap,
                       vmin=-vmax1, vmax=vmax1)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'File1 (y={y_idx})')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = axes[1, 1]
        im = ax.imshow(data2[:, y_idx, :].T, origin='lower', cmap=cmap,
                       vmin=-vmax1, vmax=vmax1)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'File2 (y={y_idx})')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = axes[1, 2]
        im = ax.imshow(diff[:, y_idx, :].T, origin='lower', cmap=cmap,
                       vmin=-vmax_diff, vmax=vmax_diff)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Difference (y={y_idx})')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        fig.suptitle(f'Comparison: {args.file1} vs {args.file2}\n'
                     f'Max diff: {np.abs(diff).max():.2e}, Corr: {corr:.6f}',
                     fontsize=11)
        fig.tight_layout()

        outfile = args.output if args.output else "compare_diff.png"
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {outfile}")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    exit(main())
