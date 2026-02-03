#!/usr/bin/env python3
"""
Verify that the output density field zoom-in region matches the config file specification.
Reads the binary density field files output by the DEBUG code.
"""

import numpy as np
import struct
import sys

def read_density_field(filename):
    """Read density field with offset information from binary file."""
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
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None


def compute_physical_region(size, offset, level):
    """Compute physical box coordinates for a grid."""
    ngrid = 2**level
    dx = 1.0 / ngrid

    x0 = offset[0] * dx
    y0 = offset[1] * dx
    z0 = offset[2] * dx

    x1 = x0 + size[0] * dx
    y1 = y0 + size[1] * dx
    z1 = z0 + size[2] * dx

    xcen = 0.5 * (x0 + x1)
    ycen = 0.5 * (y0 + y1)
    zcen = 0.5 * (z0 + z1)

    xext = x1 - x0
    yext = y1 - y0
    zext = z1 - z0

    return {
        'x0': x0, 'y0': y0, 'z0': z0,
        'x1': x1, 'y1': y1, 'z1': z1,
        'xcen': xcen, 'ycen': ycen, 'zcen': zcen,
        'xext': xext, 'yext': yext, 'zext': zext
    }


def parse_config_refinement(config_file):
    """Parse ref_center and ref_extent from config file."""
    ref_center = None
    ref_extent = None

    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('ref_center'):
                    parts = line.split('=')
                    if len(parts) == 2:
                        vals = parts[1].strip().split(',')
                        ref_center = tuple(float(v.strip()) for v in vals)
                elif line.startswith('ref_extent'):
                    parts = line.split('=')
                    if len(parts) == 2:
                        vals = parts[1].strip().split(',')
                        ref_extent = tuple(float(v.strip()) for v in vals)
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found")
        return None, None

    return ref_center, ref_extent


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_zoom_region.py <level>")
        print("Example: python verify_zoom_region.py 10")
        sys.exit(1)

    level = int(sys.argv[1])
    filename = f"delta_level{level}_real.dat"

    print(f"\n{'='*60}")
    print(f"Verifying zoom-in region for level {level}")
    print(f"{'='*60}\n")

    # Read density field
    data, size, offset = read_density_field(filename)

    if data is None:
        sys.exit(1)

    # Compute physical region
    region = compute_physical_region(size, offset, level)

    print(f"\nPhysical region in box:")
    print(f"  X: [{region['x0']:.6f}, {region['x1']:.6f}]")
    print(f"  Y: [{region['y0']:.6f}, {region['y1']:.6f}]")
    print(f"  Z: [{region['z0']:.6f}, {region['z1']:.6f}]")
    print(f"\nDerived center:")
    print(f"  ({region['xcen']:.6f}, {region['ycen']:.6f}, {region['zcen']:.6f})")
    print(f"\nDerived extent:")
    print(f"  ({region['xext']:.6f}, {region['yext']:.6f}, {region['zext']:.6f})")

    # Try to read config file
    config_file = "../ics.conf"  # Adjust path as needed
    ref_center, ref_extent = parse_config_refinement(config_file)

    if ref_center and ref_extent:
        print(f"\n{'='*60}")
        print("Comparison with config file:")
        print(f"{'='*60}")
        print(f"\nConfig ref_center: ({ref_center[0]:.6f}, {ref_center[1]:.6f}, {ref_center[2]:.6f})")
        print(f"Actual center:     ({region['xcen']:.6f}, {region['ycen']:.6f}, {region['zcen']:.6f})")

        center_diff = np.array([region['xcen'], region['ycen'], region['zcen']]) - np.array(ref_center)
        print(f"Difference:        ({center_diff[0]:.6e}, {center_diff[1]:.6e}, {center_diff[2]:.6e})")

        print(f"\nConfig ref_extent: ({ref_extent[0]:.6f}, {ref_extent[1]:.6f}, {ref_extent[2]:.6f})")
        print(f"Actual extent:     ({region['xext']:.6f}, {region['yext']:.6f}, {region['zext']:.6f})")

        extent_diff = np.array([region['xext'], region['yext'], region['zext']]) - np.array(ref_extent)
        print(f"Difference:        ({extent_diff[0]:.6e}, {extent_diff[1]:.6e}, {extent_diff[2]:.6e})")

        # Check if match is good
        center_match = np.all(np.abs(center_diff) < 1e-6)
        extent_match = np.all(np.abs(extent_diff) < 1e-6)

        print(f"\n{'='*60}")
        if center_match and extent_match:
            print("✓ VERIFICATION PASSED: Zoom-in region matches config!")
        else:
            print("✗ VERIFICATION FAILED: Zoom-in region does NOT match config!")
            if not center_match:
                print("  - Center mismatch!")
            if not extent_match:
                print("  - Extent mismatch!")
        print(f"{'='*60}\n")

    # Print basic statistics
    print(f"\nDensity field statistics:")
    print(f"  Mean:  {np.mean(data):.6e}")
    print(f"  Std:   {np.std(data):.6e}")
    print(f"  Min:   {np.min(data):.6e}")
    print(f"  Max:   {np.max(data):.6e}")


if __name__ == "__main__":
    main()
