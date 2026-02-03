#!/usr/bin/env python3
"""
Compare delta field slices from zoom and no-zoom simulations.
Plots both on the same grid size for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

def read_delta_field(filename):
    """Read density field from binary file with offset information."""
    try:
        with open(filename, 'rb') as f:
            # Read header
            nx = struct.unpack('Q', f.read(8))[0]  # size_t = 8 bytes
            ny = struct.unpack('Q', f.read(8))[0]
            nz = struct.unpack('Q', f.read(8))[0]
            ox = struct.unpack('i', f.read(4))[0]  # int = 4 bytes
            oy = struct.unpack('i', f.read(4))[0]
            oz = struct.unpack('i', f.read(4))[0]

            print(f"Reading {filename}:")
            print(f"  Grid size: {nx} × {ny} × {nz}")
            print(f"  Grid offset: ({ox}, {oy}, {oz})")

            # Read data
            data = np.fromfile(f, dtype=np.float64, count=nx*ny*nz)
            data = data.reshape((nx, ny, nz), order='C')

            return data, (nx, ny, nz), (ox, oy, oz)
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        return None, None, None


def plot_comparison(nozoom_file, zoom_file, level, slice_axis='z', slice_index=None):
    """
    Plot comparison of zoom and no-zoom delta fields.

    Parameters:
    -----------
    nozoom_file : str
        Path to no-zoom delta file
    zoom_file : str
        Path to zoom delta file
    level : int
        Refinement level
    slice_axis : str
        Axis perpendicular to slice ('x', 'y', or 'z')
    slice_index : int or None
        Index of slice. If None, use middle slice.
    """

    # Read data
    print("\n" + "="*60)
    data_nozoom, size_nozoom, offset_nozoom = read_delta_field(nozoom_file)
    if data_nozoom is None:
        return

    print()
    data_zoom, size_zoom, offset_zoom = read_delta_field(zoom_file)
    if data_zoom is None:
        return

    # Get full grid size at this level
    ngrid = 2**level
    print(f"\nFull grid size at level {level}: {ngrid}^3")

    # Determine slice axis and index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[slice_axis.lower()]

    if slice_index is None:
        # Use middle of zoom region for zoom data
        slice_index = size_zoom[axis_idx] // 2
        print(f"Using middle slice along {slice_axis}-axis: index {slice_index}")

    # Extract slices
    if axis_idx == 0:  # x-axis
        slice_nozoom = data_nozoom[slice_index, :, :]
        slice_zoom = data_zoom[slice_index, :, :] if slice_index < size_zoom[0] else None
        other_axes = (1, 2)  # y, z
    elif axis_idx == 1:  # y-axis
        slice_nozoom = data_nozoom[:, slice_index, :]
        slice_zoom = data_zoom[:, slice_index, :] if slice_index < size_zoom[1] else None
        other_axes = (0, 2)  # x, z
    else:  # z-axis
        slice_nozoom = data_nozoom[:, :, slice_index]
        slice_zoom = data_zoom[:, :, slice_index] if slice_index < size_zoom[2] else None
        other_axes = (0, 1)  # x, y

    # Create canvas for zoom data (same size as nozoom)
    canvas_zoom = np.zeros((ngrid, ngrid))

    if slice_zoom is not None:
        # Place zoom data at correct position in canvas
        # Offset might be in parent coordinates, multiply by 2^(level_diff)
        # For now, assume offset is already in level coordinates
        ox_canvas = offset_zoom[other_axes[0]]
        oy_canvas = offset_zoom[other_axes[1]]

        # If offset seems wrong (too small), might be in parent coordinates
        # Simple heuristic: if offset * 2 would make more sense, use it
        if ox_canvas < size_zoom[other_axes[0]]:
            # Offset might be in parent level coordinates
            print(f"Note: Offset might be in parent coordinates, adjusting...")
            ox_canvas = ox_canvas * 2
            oy_canvas = oy_canvas * 2

        sx = size_zoom[other_axes[0]]
        sy = size_zoom[other_axes[1]]

        print(f"\nPlacing zoom region in canvas:")
        print(f"  Canvas size: {ngrid} × {ngrid}")
        print(f"  Zoom region: {sx} × {sy}")
        print(f"  Position: ({ox_canvas}, {oy_canvas})")

        # Place zoom data (handle boundary conditions)
        x_end = min(ox_canvas + sx, ngrid)
        y_end = min(oy_canvas + sy, ngrid)
        x_start = max(0, ox_canvas)
        y_start = max(0, oy_canvas)

        dx = x_end - x_start
        dy = y_end - y_start

        if dx > 0 and dy > 0:
            canvas_zoom[x_start:x_end, y_start:y_end] = slice_zoom[:dx, :dy]
        else:
            print("Warning: Zoom region outside canvas bounds!")

    # Set up colormap (same range for both)
    vmin = min(slice_nozoom.min(), canvas_zoom[canvas_zoom != 0].min() if np.any(canvas_zoom != 0) else slice_nozoom.min())
    vmax = max(slice_nozoom.max(), canvas_zoom[canvas_zoom != 0].max() if np.any(canvas_zoom != 0) else slice_nozoom.max())

    print(f"\nColormap range: [{vmin:.4f}, {vmax:.4f}]")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot no-zoom
    im0 = axes[0].imshow(slice_nozoom.T, origin='lower', cmap='RdBu_r',
                         vmin=vmin, vmax=vmax, extent=[0, ngrid, 0, ngrid])
    axes[0].set_title(f'No-Zoom: Level {level} ({ngrid}³)\n{slice_axis}-slice at index {slice_index}',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'{"xyz"[other_axes[0]]}-axis (cells)')
    axes[0].set_ylabel(f'{"xyz"[other_axes[1]]}-axis (cells)')
    axes[0].grid(True, alpha=0.3)

    # Plot zoom on same grid
    # Use black background for empty regions
    cmap_zoom = plt.cm.RdBu_r.copy()
    cmap_zoom.set_bad(color='black')

    # Mask zeros as bad values
    canvas_zoom_masked = np.ma.masked_where(canvas_zoom == 0, canvas_zoom)

    im1 = axes[1].imshow(canvas_zoom_masked.T, origin='lower', cmap=cmap_zoom,
                         vmin=vmin, vmax=vmax, extent=[0, ngrid, 0, ngrid])
    axes[1].set_title(f'Zoom: Level {level} (zoom region on {ngrid}³ grid)\n{slice_axis}-slice at index {slice_index}',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'{"xyz"[other_axes[0]]}-axis (cells)')
    axes[1].set_ylabel(f'{"xyz"[other_axes[1]]}-axis (cells)')
    axes[1].grid(True, alpha=0.3)

    # Draw rectangle showing zoom region
    if slice_zoom is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((ox_canvas, oy_canvas), sx, sy,
                        linewidth=2, edgecolor='yellow', facecolor='none',
                        linestyle='--', label='Zoom region')
        axes[1].add_patch(rect)
        axes[1].legend()

    # Add colorbar
    fig.colorbar(im0, ax=axes, label='Density contrast δ', fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save figure
    output_name = f'comparison_level{level}_{slice_axis}slice{slice_index}.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {output_name}")

    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("Statistics:")
    print(f"  No-zoom: mean={slice_nozoom.mean():.6e}, std={slice_nozoom.std():.6e}")
    if np.any(canvas_zoom != 0):
        zoom_data = canvas_zoom[canvas_zoom != 0]
        print(f"  Zoom:    mean={zoom_data.mean():.6e}, std={zoom_data.std():.6e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_zoom_slices.py <level> [slice_axis] [slice_index]")
        print("  level: refinement level (e.g., 8)")
        print("  slice_axis: 'x', 'y', or 'z' (default: 'z')")
        print("  slice_index: slice index (default: middle of zoom region)")
        print("\nExample:")
        print("  python compare_zoom_slices.py 8")
        print("  python compare_zoom_slices.py 8 z 32")
        sys.exit(1)

    level = int(sys.argv[1])
    slice_axis = sys.argv[2] if len(sys.argv) > 2 else 'z'
    slice_index = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # File names (adjust paths as needed)
    nozoom_file = f'delta_level{level}_real_nozoom.dat'  # You need to rename your no-zoom file
    zoom_file = f'delta_level{level}_real.dat'

    print("Density Field Slice Comparison")
    print("="*60)
    print(f"Level: {level}")
    print(f"No-zoom file: {nozoom_file}")
    print(f"Zoom file: {zoom_file}")

    plot_comparison(nozoom_file, zoom_file, level, slice_axis, slice_index)


if __name__ == "__main__":
    main()
