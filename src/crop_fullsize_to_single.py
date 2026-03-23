#!/usr/bin/env python3
"""
Crop fullsize delta field to match the single zoom-in region and compare.

Usage:
    # Compare fullsize with two single fields
    python crop_fullsize_to_single.py --fullsize fullsize/delta_level8_real.dat \
        --single1 single/delta_level8_real.dat \
        --single2 merge/cube0/delta_level8_real.dat \
        --plot

    # Specify custom refinement region (default: [0.375:0.625, 0.375:0.625, 0:1])
    python crop_fullsize_to_single.py --fullsize ... --single1 ... --single2 ... \
        --ref-left 0.375,0.375,0 --ref-right 0.625,0.625,1
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


def write_field(filename, data, offset):
    """Write a MUSIC2 debug binary field."""
    nx, ny, nz = data.shape
    ox, oy, oz = offset

    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', nx))
        f.write(struct.pack('Q', ny))
        f.write(struct.pack('Q', nz))
        f.write(struct.pack('i', ox))
        f.write(struct.pack('i', oy))
        f.write(struct.pack('i', oz))
        data.astype(np.float64).tofile(f)


def print_field_info(name, data, size, offset):
    """Print field information."""
    print(f"  {name}:")
    print(f"    Size: {size[0]} x {size[1]} x {size[2]}")
    print(f"    Offset: {offset}")
    print(f"    Range: [{data.min():.6e}, {data.max():.6e}]")
    print(f"    Mean: {data.mean():.6e}, Std: {data.std():.6e}")


def compute_diff_stats(data1, data2, name1, name2):
    """Compute and print difference statistics."""
    diff = data1 - data2

    print(f"\n  {name1} vs {name2}:")
    print(f"    Max |diff|: {np.abs(diff).max():.6e}")
    print(f"    RMS diff:   {np.sqrt((diff**2).mean()):.6e}")

    corr = np.corrcoef(data1.flatten(), data2.flatten())[0, 1]
    print(f"    Correlation: {corr:.10f}")

    return diff, corr


def main():
    parser = argparse.ArgumentParser(
        description="Crop fullsize and compare with two single zoom-in fields")
    parser.add_argument("--fullsize", required=True,
                        help="Path to fullsize delta field")
    parser.add_argument("--single1", required=True,
                        help="Path to first single delta field")
    parser.add_argument("--single2", required=True,
                        help="Path to second single delta field")
    parser.add_argument("--ref-left", default="0.375,0.375,0",
                        help="Refinement region left corner (default: 0.375,0.375,0)")
    parser.add_argument("--ref-right", default="0.625,0.625,1",
                        help="Refinement region right corner (default: 0.625,0.625,1)")
    parser.add_argument("--output", "-o", default="fullsize_cropped.dat",
                        help="Output cropped field filename")
    parser.add_argument("--plot", action="store_true",
                        help="Plot comparison slices")
    parser.add_argument("--plot-output", default="crop_comparison.png",
                        help="Output plot filename")
    parser.add_argument("--slice-z", type=int, default=None,
                        help="Z index for x-y slice (default: middle)")
    parser.add_argument("--slice-y", type=int, default=None,
                        help="Y index for x-z slice (default: middle)")
    parser.add_argument("--plane", choices=["xy", "xz", "both"], default="xy",
                        help="Which plane to plot: xy, xz, or both (default: xy)")
    args = parser.parse_args()

    # Parse refinement region
    ref_left = [float(x) for x in args.ref_left.split(',')]
    ref_right = [float(x) for x in args.ref_right.split(',')]

    # Read fullsize field
    print("Reading fields...")
    fullsize_data, fullsize_size, fullsize_offset = read_field(args.fullsize)
    print_field_info("Fullsize", fullsize_data, fullsize_size, fullsize_offset)

    # Read single fields
    single1_data, single1_size, single1_offset = read_field(args.single1)
    print_field_info("Single1", single1_data, single1_size, single1_offset)

    single2_data, single2_size, single2_offset = read_field(args.single2)
    print_field_info("Single2", single2_data, single2_size, single2_offset)

    # Calculate crop region
    x_start = int(ref_left[0] * fullsize_size[0])
    y_start = int(ref_left[1] * fullsize_size[1])
    z_start = int(ref_left[2] * fullsize_size[2])
    x_end = int(ref_right[0] * fullsize_size[0])
    y_end = int(ref_right[1] * fullsize_size[1])
    z_end = int(ref_right[2] * fullsize_size[2])

    print(f"\nCrop region: [{x_start}:{x_end}, {y_start}:{y_end}, {z_start}:{z_end}]")
    print(f"  Physical: [{ref_left[0]}:{ref_right[0]}, {ref_left[1]}:{ref_right[1]}, {ref_left[2]}:{ref_right[2]}]")

    # Crop fullsize
    cropped_data = fullsize_data[x_start:x_end, y_start:y_end, z_start:z_end].copy()
    print(f"  Cropped size: {cropped_data.shape[0]} x {cropped_data.shape[1]} x {cropped_data.shape[2]}")

    # Check size compatibility
    if cropped_data.shape != single1_data.shape:
        print(f"\nWARNING: Shape mismatch with single1!")
        print(f"  Cropped: {cropped_data.shape}, Single1: {single1_data.shape}")
    if cropped_data.shape != single2_data.shape:
        print(f"\nWARNING: Shape mismatch with single2!")
        print(f"  Cropped: {cropped_data.shape}, Single2: {single2_data.shape}")
    if single1_data.shape != single2_data.shape:
        print(f"\nWARNING: Shape mismatch between single1 and single2!")
        print(f"  Single1: {single1_data.shape}, Single2: {single2_data.shape}")

    # Save cropped field
    write_field(args.output, cropped_data, (x_start, y_start, z_start))
    print(f"\nWrote cropped field to {args.output}")

    # Compute differences
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)

    diff_fs_s1, corr_fs_s1 = compute_diff_stats(cropped_data, single1_data, "Fullsize", "Single1")
    diff_fs_s2, corr_fs_s2 = compute_diff_stats(cropped_data, single2_data, "Fullsize", "Single2")
    diff_s1_s2, corr_s1_s2 = compute_diff_stats(single1_data, single2_data, "Single1", "Single2")

    # Plot
    if args.plot:
        import matplotlib.pyplot as plt

        # Determine slice indices
        nx, ny, nz = cropped_data.shape
        z_idx = args.slice_z if args.slice_z is not None else nz // 2
        y_idx = args.slice_y if args.slice_y is not None else ny // 2

        # Color scale for data
        vmax_data = max(
            abs(cropped_data.min()), abs(cropped_data.max()),
            abs(single1_data.min()), abs(single1_data.max()),
            abs(single2_data.min()), abs(single2_data.max())
        )

        # Color scale for differences
        vmax_diff = max(
            abs(diff_fs_s1.min()), abs(diff_fs_s1.max()),
            abs(diff_fs_s2.min()), abs(diff_fs_s2.max()),
            abs(diff_s1_s2.min()), abs(diff_s1_s2.max())
        )
        if vmax_diff == 0:
            vmax_diff = 1e-10

        cmap = plt.cm.RdBu_r

        if args.plane == "both":
            # Plot both xy and xz planes (4 rows)
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))

            # Row 0: x-y slices (data)
            for col, (data, name) in enumerate([
                (cropped_data, "Fullsize"),
                (single1_data, "Single1"),
                (single2_data, "Single2")
            ]):
                ax = axes[0, col]
                im = ax.imshow(data[:, :, z_idx].T, origin='lower', cmap=cmap,
                               vmin=-vmax_data, vmax=vmax_data, aspect='equal')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{name} (z={z_idx})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            # Row 1: x-y slices (differences)
            for col, (diff, corr, name) in enumerate([
                (diff_fs_s1, corr_fs_s1, "FS - S1"),
                (diff_fs_s2, corr_fs_s2, "FS - S2"),
                (diff_s1_s2, corr_s1_s2, "S1 - S2")
            ]):
                ax = axes[1, col]
                im = ax.imshow(diff[:, :, z_idx].T, origin='lower', cmap=cmap,
                               vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{name} (corr={corr:.4f})')
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            # Row 2: x-z slices (data)
            for col, (data, name) in enumerate([
                (cropped_data, "Fullsize"),
                (single1_data, "Single1"),
                (single2_data, "Single2")
            ]):
                ax = axes[2, col]
                im = ax.imshow(data[:, y_idx, :].T, origin='lower', cmap=cmap,
                               vmin=-vmax_data, vmax=vmax_data, aspect='equal')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{name} (y={y_idx})')
                ax.set_xlabel('x')
                ax.set_ylabel('z')

            # Row 3: x-z slices (differences)
            for col, (diff, corr, name) in enumerate([
                (diff_fs_s1, corr_fs_s1, "FS - S1"),
                (diff_fs_s2, corr_fs_s2, "FS - S2"),
                (diff_s1_s2, corr_s1_s2, "S1 - S2")
            ]):
                ax = axes[3, col]
                im = ax.imshow(diff[:, y_idx, :].T, origin='lower', cmap=cmap,
                               vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{name} (corr={corr:.4f})')
                ax.set_xlabel('x')
                ax.set_ylabel('z')

            fig.suptitle(f'Comparison: xy at z={z_idx}, xz at y={y_idx}\n'
                         f'Max |diff|: FS-S1={np.abs(diff_fs_s1).max():.2e}, '
                         f'FS-S2={np.abs(diff_fs_s2).max():.2e}, '
                         f'S1-S2={np.abs(diff_s1_s2).max():.2e}',
                         fontsize=11)

        else:
            # Plot single plane (2 rows)
            fig, axes = plt.subplots(2, 3, figsize=(15, 12))

            if args.plane == "xy":
                slice_idx = z_idx
                slice_label = f"z={z_idx}"
                xlabel, ylabel = 'x', 'y'
                get_slice = lambda d: d[:, :, z_idx].T
            else:  # xz
                slice_idx = y_idx
                slice_label = f"y={y_idx}"
                xlabel, ylabel = 'x', 'z'
                get_slice = lambda d: d[:, y_idx, :].T

            # Top row: data
            for col, (data, name) in enumerate([
                (cropped_data, "Fullsize"),
                (single1_data, "Single1"),
                (single2_data, "Single2")
            ]):
                ax = axes[0, col]
                im = ax.imshow(get_slice(data), origin='lower', cmap=cmap,
                               vmin=-vmax_data, vmax=vmax_data, aspect='equal')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{name} ({slice_label})')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            # Bottom row: differences
            for col, (diff, corr, name) in enumerate([
                (diff_fs_s1, corr_fs_s1, "FS - S1"),
                (diff_fs_s2, corr_fs_s2, "FS - S2"),
                (diff_s1_s2, corr_s1_s2, "S1 - S2")
            ]):
                ax = axes[1, col]
                im = ax.imshow(get_slice(diff), origin='lower', cmap=cmap,
                               vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{name} (corr={corr:.4f})')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            fig.suptitle(f'Comparison at {slice_label}\n'
                         f'Max |diff|: FS-S1={np.abs(diff_fs_s1).max():.2e}, '
                         f'FS-S2={np.abs(diff_fs_s2).max():.2e}, '
                         f'S1-S2={np.abs(diff_s1_s2).max():.2e}',
                         fontsize=11)

        fig.tight_layout()
        fig.savefig(args.plot_output, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {args.plot_output}")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    exit(main())
