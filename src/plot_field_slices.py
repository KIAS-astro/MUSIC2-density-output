#!/usr/bin/env python3
"""
Plot delta and theta field slices from MUSIC2 zoom-in ICs.

For each refinement level, plots a 1-cell-thick slice in both the x-y plane
(constant z) and the x-z plane (constant y). Higher levels cover smaller
subregions; the empty area outside the subgrid is left white.

Usage:
    python plot_field_slices.py [options]

Examples:
    python plot_field_slices.py
    python plot_field_slices.py --dir /path/to/output
    python plot_field_slices.py --slice-z 0.5 --slice-y 0.5
    python plot_field_slices.py --fields delta
    python plot_field_slices.py --levels 6 7 8
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import struct
import glob
import os
import re
import argparse


def read_field(filename):
    """Read a MUSIC2 debug binary field (delta or theta).

    Header: nx(uint64) ny(uint64) nz(uint64) ox(int32) oy(int32) oz(int32)
    Data:   float64[nx*ny*nz] in C order (i,j,k) with k fastest

    Note: the offset stored in the file is relative to the parent level
    in parent-level cell units, NOT the absolute offset at the current level.
    Use compute_absolute_offsets() to convert.
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


def detect_levels(data_dir, prefix):
    """Find all available levels for a given field prefix (delta or theta)."""
    pattern = os.path.join(data_dir, f"{prefix}_level*_real.dat")
    files = sorted(glob.glob(pattern))
    levels = []
    for f in files:
        m = re.search(rf"{prefix}_level(\d+)_real\.dat$", f)
        if m:
            levels.append(int(m.group(1)))
    return sorted(levels)


def compute_absolute_offsets(data_dir, prefix, levels):
    """Compute absolute offsets in current-level cell units for each level.

    The binary files store offsets relative to the parent level in parent-level
    cell units.  The absolute offset follows the GridHierarchy formula:
        offset_abs[level] = 2 * (offset_abs[level-1] + offset[level])
    with offset_abs[levelmin] = (0, 0, 0) for the base level.
    """
    # Read raw (parent-relative) offsets for every level
    raw_offsets = {}
    for level in levels:
        fname = os.path.join(data_dir, f"{prefix}_level{level}_real.dat")
        _, _, offset = read_field(fname)
        raw_offsets[level] = offset

    levelmin = levels[0]
    abs_offsets = {levelmin: (0, 0, 0)}

    for i in range(1, len(levels)):
        level = levels[i]
        prev = levels[i - 1]
        parent_abs = abs_offsets[prev]
        raw = raw_offsets[level]
        abs_offsets[level] = tuple(
            2 * (parent_abs[d] + raw[d]) for d in range(3)
        )

    return abs_offsets


def make_canvas(data, size, offset, ngrid, axis, slice_idx):
    """Extract a 1-cell slice and embed it in a full-box canvas.

    Parameters
    ----------
    data : ndarray (nx, ny, nz)
    size : (nx, ny, nz)
    offset : (ox, oy, oz)
    ngrid : int  -- 2**level, the full box grid count
    axis : int   -- axis perpendicular to the slice (0=x, 1=y, 2=z)
    slice_idx : int -- index in the *full box* grid along that axis

    Returns
    -------
    canvas : 2-D ndarray (ngrid, ngrid) filled with NaN outside the subgrid
    valid  : bool -- whether the requested slice intersects the subgrid
    """
    local_idx = slice_idx - offset[axis]

    if local_idx < 0 or local_idx >= size[axis]:
        return np.full((ngrid, ngrid), np.nan), False

    # The two axes that span the 2-D slice (in order)
    axes_2d = [a for a in range(3) if a != axis]
    a0, a1 = axes_2d

    # Extract the slice from the 3-D array
    slicer = [slice(None)] * 3
    slicer[axis] = local_idx
    plane = data[tuple(slicer)]  # shape (size[a0], size[a1])

    # Embed into canvas
    canvas = np.full((ngrid, ngrid), np.nan)
    i0 = offset[a0]
    j0 = offset[a1]
    canvas[i0:i0 + size[a0], j0:j0 + size[a1]] = plane

    return canvas, True


def fractional_to_index(frac, ngrid):
    """Convert a fractional box coordinate [0,1) to a grid index."""
    idx = int(frac * ngrid)
    return max(0, min(idx, ngrid - 1))


def plot_levels(data_dir, prefix, levels, slice_z_frac, slice_y_frac, outname):
    """Create the multi-level slice figure for one field.

    Two rows: top = x-y plane (z-slice), bottom = x-z plane (y-slice).
    One column per level.
    """
    n_levels = len(levels)
    fig, axes = plt.subplots(2, n_levels, figsize=(5 * n_levels, 10),
                             squeeze=False)

    field_label = r"$\delta$" if prefix == "delta" else r"$\theta$"

    # Compute absolute offsets from the parent-relative ones in the files
    abs_offsets = compute_absolute_offsets(data_dir, prefix, levels)

    # First pass: read all data and compute global colour limits
    vmin_global = np.inf
    vmax_global = -np.inf

    canvas_cache = {}
    for level in levels:
        fname = os.path.join(data_dir, f"{prefix}_level{level}_real.dat")
        data, size, _ = read_field(fname)
        offset = abs_offsets[level]
        ngrid = 2 ** level

        z_idx = fractional_to_index(slice_z_frac, ngrid)
        canvas_xy, valid_xy = make_canvas(data, size, offset, ngrid,
                                          axis=2, slice_idx=z_idx)

        y_idx = fractional_to_index(slice_y_frac, ngrid)
        canvas_xz, valid_xz = make_canvas(data, size, offset, ngrid,
                                          axis=1, slice_idx=y_idx)

        canvas_cache[level] = {
            'xy': canvas_xy, 'valid_xy': valid_xy, 'z_idx': z_idx,
            'xz': canvas_xz, 'valid_xz': valid_xz, 'y_idx': y_idx,
            'ngrid': ngrid, 'size': size, 'offset': offset,
        }

        for c in (canvas_xy, canvas_xz):
            vals = c[np.isfinite(c)]
            if vals.size > 0:
                vmin_global = min(vmin_global, vals.min())
                vmax_global = max(vmax_global, vals.max())

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='white')

    for col, level in enumerate(levels):
        info = canvas_cache[level]
        ngrid = info['ngrid']
        dx = 1.0 / ngrid

        # Per-level symmetric colour scale (so each level's structure is visible)
        vlim_xy = 0.0
        vlim_xz = 0.0
        vals_xy = info['xy'][np.isfinite(info['xy'])]
        vals_xz = info['xz'][np.isfinite(info['xz'])]
        if vals_xy.size > 0:
            vlim_xy = max(abs(vals_xy.min()), abs(vals_xy.max()))
        if vals_xz.size > 0:
            vlim_xz = max(abs(vals_xz.min()), abs(vals_xz.max()))
        # Use the same scale for both slices within a level
        vlim = max(vlim_xy, vlim_xz) if max(vlim_xy, vlim_xz) > 0 else 1.0

        # ---- x-y plane (top row) ----
        ax = axes[0, col]
        im_xy = ax.imshow(info['xy'].T, origin='lower', cmap=cmap,
                          vmin=-vlim, vmax=vlim, extent=[0, 1, 0, 1],
                          aspect='equal', interpolation='none')
        plt.colorbar(im_xy, ax=ax, fraction=0.046, pad=0.04, label=field_label)
        # Subgrid boundary
        x0 = info['offset'][0] * dx
        y0 = info['offset'][1] * dx
        sx = info['size'][0] * dx
        sy = info['size'][1] * dx
        rect = Rectangle((x0, y0), sx, sy, linewidth=1.5,
                          edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        status = "" if info['valid_xy'] else "  [no data]"
        ax.set_title(f"Level {level}  ({ngrid}$^3$)\n"
                     f"x-y slice @ z={info['z_idx']}/{ngrid}"
                     f"{status}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # ---- x-z plane (bottom row) ----
        ax = axes[1, col]
        im_xz = ax.imshow(info['xz'].T, origin='lower', cmap=cmap,
                          vmin=-vlim, vmax=vlim, extent=[0, 1, 0, 1],
                          aspect='equal', interpolation='none')
        plt.colorbar(im_xz, ax=ax, fraction=0.046, pad=0.04, label=field_label)
        x0 = info['offset'][0] * dx
        z0 = info['offset'][2] * dx
        sx = info['size'][0] * dx
        sz = info['size'][2] * dx
        rect = Rectangle((x0, z0), sx, sz, linewidth=1.5,
                          edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        status = "" if info['valid_xz'] else "  [no data]"
        ax.set_title(f"Level {level}  ({ngrid}$^3$)\n"
                     f"x-z slice @ y={info['y_idx']}/{ngrid}"
                     f"{status}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    fig.suptitle(f"{field_label} field  --  1-cell-thick slices", fontsize=14,
                 fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1.0, 0.95])
    fig.savefig(outname, dpi=200, bbox_inches='tight')
    print(f"Saved {outname}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot MUSIC2 delta/theta slices for all zoom levels.")
    parser.add_argument("--dir", default=".",
                        help="Directory containing the *_level*_real.dat files")
    parser.add_argument("--fields", nargs="+", default=["delta", "theta"],
                        choices=["delta", "theta"],
                        help="Which fields to plot (default: both)")
    parser.add_argument("--levels", nargs="+", type=int, default=None,
                        help="Levels to plot (default: auto-detect)")
    parser.add_argument("--slice-z", type=float, default=0.5, dest="slice_z",
                        help="Fractional z-position for x-y slice (0-1, "
                             "default 0.5)")
    parser.add_argument("--slice-y", type=float, default=0.5, dest="slice_y",
                        help="Fractional y-position for x-z slice (0-1, "
                             "default 0.5)")
    parser.add_argument("--out-prefix", default="slice", dest="out_prefix",
                        help="Prefix for output PNG files (default: slice)")
    args = parser.parse_args()

    for prefix in args.fields:
        levels = args.levels
        if levels is None:
            levels = detect_levels(args.dir, prefix)
        if not levels:
            print(f"No {prefix}_level*_real.dat files found in {args.dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Plotting {prefix} field for levels {levels}")
        print(f"  z-slice fraction: {args.slice_z}")
        print(f"  y-slice fraction: {args.slice_y}")
        print(f"{'='*60}")

        abs_offsets = compute_absolute_offsets(args.dir, prefix, levels)
        for level in levels:
            fname = os.path.join(args.dir, f"{prefix}_level{level}_real.dat")
            _, size, raw_off = read_field(fname)
            off = abs_offsets[level]
            ngrid = 2 ** level
            dx = 1.0 / ngrid
            print(f"  Level {level}: grid {size[0]}x{size[1]}x{size[2]}, "
                  f"raw_offset ({raw_off[0]},{raw_off[1]},{raw_off[2]}), "
                  f"abs_offset ({off[0]},{off[1]},{off[2]}), "
                  f"phys [{off[0]*dx:.4f}:{(off[0]+size[0])*dx:.4f}, "
                  f"{off[1]*dx:.4f}:{(off[1]+size[1])*dx:.4f}, "
                  f"{off[2]*dx:.4f}:{(off[2]+size[2])*dx:.4f}]")

        outname = f"{args.out_prefix}_{prefix}.png"
        plot_levels(args.dir, prefix, levels,
                    args.slice_z, args.slice_y, outname)


if __name__ == "__main__":
    main()
