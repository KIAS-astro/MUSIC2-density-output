#!/usr/bin/env python3
"""
Merge delta field blocks from multiple cube directories along the z direction.

Each cube directory (cube00, cube01, ...) contains delta_level{N}_real.dat files.
This script reads a specified level from each cube and concatenates them in z order.

Usage:
    python merge_cubes_z.py --level 8
    python merge_cubes_z.py --level 8 --cubes cube00 cube01 cube02
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

        # Write data in C order
        data.astype(np.float64).tofile(f)

    print(f"Wrote {filename}")
    print(f"  Grid size: {nx} x {ny} x {nz}")
    print(f"  Offset: ({ox}, {oy}, {oz})")


def find_cube_dirs(base_dir, pattern="cube*"):
    """Find and sort cube directories."""
    dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
    # Filter to only directories
    dirs = [d for d in dirs if os.path.isdir(d)]
    return dirs


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally (cube2 < cube10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def main():
    parser = argparse.ArgumentParser(
        description="Merge delta blocks from cube directories along z direction.")
    parser.add_argument("--level", type=int, required=True,
                        help="Level to merge (e.g., 8)")
    parser.add_argument("--cubes", nargs="+", default=None,
                        help="Cube directories to merge in order. "
                             "If not specified, auto-detects cube* directories.")
    parser.add_argument("--dir", default=".",
                        help="Base directory containing cube subdirectories")
    parser.add_argument("--prefix", default="delta",
                        help="Field prefix (default: delta)")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: merged_{prefix}_level{N}.dat)")
    args = parser.parse_args()

    # Find cube directories
    if args.cubes:
        cube_dirs = [os.path.join(args.dir, c) if not os.path.isabs(c) else c
                     for c in args.cubes]
    else:
        cube_dirs = find_cube_dirs(args.dir)
        cube_dirs = sorted(cube_dirs, key=lambda x: natural_sort_key(os.path.basename(x)))

    if not cube_dirs:
        print(f"No cube directories found in {args.dir}")
        return 1

    print(f"Merging level {args.level} from {len(cube_dirs)} cube directories:")
    for d in cube_dirs:
        print(f"  {d}")
    print()

    # Read all blocks
    blocks = []
    sizes = []
    offsets = []

    for cube_dir in cube_dirs:
        filename = os.path.join(cube_dir, f"{args.prefix}_level{args.level}_real.dat")
        if not os.path.exists(filename):
            print(f"Error: {filename} not found")
            return 1

        data, size, offset = read_field(filename)
        blocks.append(data)
        sizes.append(size)
        offsets.append(offset)

        print(f"Read {filename}")
        print(f"  Size: {size[0]} x {size[1]} x {size[2]}")
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

    # Concatenate along z axis
    print(f"\nConcatenating {len(blocks)} blocks along z axis...")
    merged = np.concatenate(blocks, axis=2)

    # Compute merged offset (use first block's offset, z offset is from first block)
    merged_offset = (ox_ref, oy_ref, offsets[0][2])

    # Total z size
    total_nz = sum(s[2] for s in sizes)
    print(f"Merged size: {nx_ref} x {ny_ref} x {total_nz}")
    print(f"Merged offset: ({merged_offset[0]}, {merged_offset[1]}, {merged_offset[2]})")

    # Compute statistics
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
