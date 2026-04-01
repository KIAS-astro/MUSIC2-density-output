#!/usr/bin/env python3
"""
Merge delta field blocks from multiple cube directories along the z direction.

Each cube directory (cube00, cube01, ...) contains delta_level{N}_real.dat files.
This script reads a specified level from each cube and concatenates them in z order.

When --padding is specified, each cube's overlap region is trimmed before merging:
  - First cube:  keep [:, :, : nz - cut]          (right edge trimmed)
  - Middle cubes: keep [:, :, cut : nz - cut]      (both edges trimmed)
  - Last cube:   keep [:, :, cut :]                (left edge trimmed)
where cut = cube_padding_cells // 2.

The z offset in the output header is updated to reflect the trimmed first block.

Usage:
    python merge_cubes_z.py --level 8
    python merge_cubes_z.py --level 8 --cubes cube00 cube01 cube02
    python merge_cubes_z.py --level 8 --padding 16
    python merge_cubes_z.py --level 8 --prefix delta --output merged_delta_level8.dat
"""

import numpy as np
import struct
import glob
import os
import re
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
    """Write a MUSIC2 debug binary field.

    Header: nx(uint64) ny(uint64) nz(uint64) ox(int32) oy(int32) oz(int32)
    Data:   float64[nx*ny*nz] in C order (i,j,k) with k fastest
    """
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

    print(f"Wrote {filename}")
    print(f"  Grid size: {nx} x {ny} x {nz}")
    print(f"  Offset: ({ox}, {oy}, {oz})")


def find_cube_dirs(base_dir, pattern="cube*"):
    """Find and sort cube directories."""
    dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
    dirs = [d for d in dirs if os.path.isdir(d)]
    return dirs


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally (cube2 < cube10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def trim_block(data, cut, is_first, is_last):
    """
    Trim padding/2 grid cells from the z edges of a block.

    Parameters:
    -----------
    data : ndarray  shape (nx, ny, nz)
    cut  : int      number of z cells to remove from each internal edge
    is_first : bool  do not trim the left (low-z) edge
    is_last  : bool  do not trim the right (high-z) edge

    Returns:
    --------
    trimmed : ndarray
    z_start : int   index of first kept z slice in original block
                    (used to update the oz offset)
    """
    nz = data.shape[2]
    z_lo = 0    if is_first else cut
    z_hi = nz   if is_last  else nz - cut
    return data[:, :, z_lo:z_hi], z_lo


def main():
    parser = argparse.ArgumentParser(
        description="Merge delta blocks from cube directories along z direction.")
    parser.add_argument("--level", type=int, required=True,
                        help="Level to merge (e.g., 8)")
    parser.add_argument("--cubes", nargs="+", default=None,
                        help="Cube directories to merge in order (left to right). "
                             "If not specified, auto-detects cube* directories.")
    parser.add_argument("--dir", default=".",
                        help="Base directory containing cube subdirectories")
    parser.add_argument("--prefix", default="delta",
                        help="Field prefix (default: delta)")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: merged_{prefix}_level{N}.dat)")
    parser.add_argument("--padding", type=int, default=0,
                        dest="padding",
                        help="cube_padding_cells used when generating ICs (default: 0). "
                             "Must be even. cut = padding // 2 cells are removed from "
                             "each internal edge.")
    args = parser.parse_args()

    # Validate padding
    if args.padding < 0:
        print("Error: --padding must be >= 0")
        return 1
    if args.padding % 2 != 0:
        print(f"Error: --padding ({args.padding}) must be even")
        return 1
    cut = args.padding // 2

    # Find cube directories
    if args.cubes:
        cube_dirs = [os.path.join(args.dir, c) if not os.path.isabs(c) else c
                     for c in args.cubes]
    else:
        cube_dirs = find_cube_dirs(args.dir)
        cube_dirs = sorted(cube_dirs,
                           key=lambda x: natural_sort_key(os.path.basename(x)))

    if not cube_dirs:
        print(f"No cube directories found in {args.dir}")
        return 1

    n_cubes = len(cube_dirs)
    print(f"Merging level {args.level} from {n_cubes} cube directories:")
    for d in cube_dirs:
        print(f"  {d}")
    if cut > 0:
        print(f"\nPadding trim: cube_padding_cells={args.padding}, cut={cut} cells per internal edge")
    print()

    # Read all blocks
    blocks = []
    sizes = []
    offsets = []

    for cube_dir in cube_dirs:
        filename = os.path.join(cube_dir,
                                f"{args.prefix}_level{args.level}_real.dat")
        if not os.path.exists(filename):
            print(f"Error: {filename} not found")
            return 1

        data, size, offset = read_field(filename)
        blocks.append(data)
        sizes.append(size)
        offsets.append(offset)

        print(f"Read {filename}")
        print(f"  Size:   {size[0]} x {size[1]} x {size[2]}")
        print(f"  Offset: ({offset[0]}, {offset[1]}, {offset[2]})")

    # Verify all blocks have same x and y dimensions
    nx_ref, ny_ref = sizes[0][0], sizes[0][1]
    ox_ref, oy_ref = offsets[0][0], offsets[0][1]

    for i, (size, offset) in enumerate(zip(sizes, offsets)):
        if size[0] != nx_ref or size[1] != ny_ref:
            print(f"Error: Block {i} has different x/y dimensions: "
                  f"{size[0]}x{size[1]} vs {nx_ref}x{ny_ref}")
            return 1
        if offset[0] != ox_ref or offset[1] != oy_ref:
            print(f"Warning: Block {i} has different x/y offset: "
                  f"({offset[0]},{offset[1]}) vs ({ox_ref},{oy_ref})")

    # Validate cut does not exceed any block's nz
    if cut > 0:
        for i, size in enumerate(sizes):
            nz = size[2]
            is_first = (i == 0)
            is_last  = (i == n_cubes - 1)
            needed = (0 if is_first else cut) + (0 if is_last else cut)
            if needed >= nz:
                print(f"Error: block {i} has nz={nz} but needs to trim "
                      f"{needed} cells total — increase cube_size or reduce padding")
                return 1

    # Trim and concatenate
    print(f"\n{'Trimming and c' if cut > 0 else 'C'}oncatenating "
          f"{n_cubes} blocks along z axis...")

    trimmed_blocks = []
    oz_first_trimmed = offsets[0][2]   # will be updated if first block is trimmed

    for i, (data, offset) in enumerate(zip(blocks, offsets)):
        is_first = (i == 0)
        is_last  = (i == n_cubes - 1)

        if cut > 0:
            trimmed, z_start = trim_block(data, cut, is_first, is_last)
            nz_before = data.shape[2]
            nz_after  = trimmed.shape[2]

            tag = ""
            if is_first:
                tag = " [leftmost — left edge kept]"
            elif is_last:
                tag = " [rightmost — right edge kept]"

            print(f"  Block {i}{tag}:")
            print(f"    raw nz = {nz_before}  →  trimmed nz = {nz_after}  "
                  f"(z_slice [{z_start}, {z_start + nz_after}])")

            # Update oz for the first block if its left edge is trimmed
            # (never happens for first block, but kept for clarity)
            if is_first:
                oz_first_trimmed = offset[2] + z_start
        else:
            trimmed = data

        trimmed_blocks.append(trimmed)

    merged = np.concatenate(trimmed_blocks, axis=2)

    # Merged z offset: first block's oz (adjusted if left edge was trimmed,
    # which never happens for the first block in practice)
    merged_offset = (ox_ref, oy_ref, oz_first_trimmed)

    total_nz = merged.shape[2]
    print(f"\nMerged size:   {nx_ref} x {ny_ref} x {total_nz}")
    print(f"Merged offset: ({merged_offset[0]}, {merged_offset[1]}, {merged_offset[2]})")

    # Verify z continuity: each trimmed block's nz should sum correctly
    if cut > 0:
        expected_nz = sum(
            s[2] - (0 if i == 0 else cut) - (0 if i == n_cubes - 1 else cut)
            for i, s in enumerate(sizes)
        )
        if total_nz != expected_nz:
            print(f"WARNING: expected merged nz={expected_nz}, got {total_nz}")
        else:
            print(f"✓ Merged nz matches expected ({expected_nz})")

    # Statistics
    print(f"\nMerged field statistics:")
    print(f"  Mean: {merged.mean():.6e}")
    print(f"  Std:  {merged.std():.6e}")
    print(f"  Min:  {merged.min():.6e}")
    print(f"  Max:  {merged.max():.6e}")

    # Write output
    if args.output:
        outfile = args.output
    else:
        outfile = f"merged_{args.prefix}_level{args.level}.dat"

    write_field(outfile, merged, merged_offset)

    return 0


if __name__ == "__main__":
    exit(main())
