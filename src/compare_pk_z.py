#!/usr/bin/env python3
"""
Compare power spectra along z direction from two MUSIC2 .dat fields.

Plots P(kz) from both files on the same figure for easy comparison.
Also shows the ratio and difference.

Usage:
    python compare_pk_z.py file1.dat file2.dat --Lz 100.0
    python compare_pk_z.py slab.dat merged.dat --Lz 100 --cube-size 33.3
    python compare_pk_z.py file1.dat file2.dat --Lz 100 --labels "Slab" "Merged"
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import argparse


def read_field(filename):
    """Read a MUSIC2 debug binary field."""
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


def compute_pk_z(data, Lz):
    """Compute 1D power spectrum along z direction."""
    nx, ny, nz = data.shape
    delta_kz = np.fft.rfft(data, axis=2)
    power = np.abs(delta_kz) ** 2
    Pk = power.mean(axis=(0, 1))
    Pk /= nz ** 2
    Pk *= Lz
    kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi
    return kz, Pk


def compute_pk_3d(data, Lx, Ly, Lz):
    """Compute full 3D power spectrum P(k), binned by |k|."""
    nx, ny, nz = data.shape
    delta_k = np.fft.rfftn(data)

    kx = np.fft.fftfreq(nx, d=Lx / nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=Ly / ny) * 2 * np.pi
    kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    power = np.abs(delta_k) ** 2
    power /= (nx * ny * nz) ** 2
    V = Lx * Ly * Lz
    power *= V

    k_fund = 2 * np.pi / max(Lx, Ly, Lz)
    k_bins = np.arange(0, k_mag.max() + k_fund, k_fund)

    Pk = np.zeros(len(k_bins) - 1)
    Nmodes = np.zeros(len(k_bins) - 1, dtype=int)

    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if mask.any():
            Pk[i] = power[mask].mean()
            Nmodes[i] = mask.sum()

    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    valid = Nmodes > 0
    return k_centers[valid], Pk[valid], Nmodes[valid]


def main():
    parser = argparse.ArgumentParser(
        description="Compare power spectra from two .dat files.")
    parser.add_argument("file1", help="First .dat file")
    parser.add_argument("file2", help="Second .dat file")
    parser.add_argument("--Lx", type=float, default=None,
                        help="Box length in x [Mpc/h]")
    parser.add_argument("--Ly", type=float, default=None,
                        help="Box length in y [Mpc/h]")
    parser.add_argument("--Lz", type=float, required=True,
                        help="Box length in z [Mpc/h]")
    parser.add_argument("--labels", nargs=2, default=None,
                        metavar=("LABEL1", "LABEL2"),
                        help="Labels for the two files")
    parser.add_argument("--cube-size", type=float, default=None, dest="cube_size",
                        help="Size of each merged cube [Mpc/h]")
    parser.add_argument("--n-cubes", type=int, default=None, dest="n_cubes",
                        help="Number of cubes merged")
    parser.add_argument("--full-3d", action="store_true", dest="full_3d",
                        help="Also compare full 3D P(k)")
    parser.add_argument("--output", "-o", default="compare_pk.png",
                        help="Output plot filename")
    args = parser.parse_args()

    Lx = args.Lx if args.Lx is not None else args.Lz
    Ly = args.Ly if args.Ly is not None else args.Lz
    Lz = args.Lz

    # Labels
    if args.labels:
        label1, label2 = args.labels
    else:
        label1 = args.file1
        label2 = args.file2

    # Read files
    print(f"Reading {args.file1}...")
    data1, size1, offset1 = read_field(args.file1)
    print(f"  Size: {size1[0]} x {size1[1]} x {size1[2]}")

    print(f"Reading {args.file2}...")
    data2, size2, offset2 = read_field(args.file2)
    print(f"  Size: {size2[0]} x {size2[1]} x {size2[2]}")

    # Compute P(kz)
    print("\nComputing P(kz)...")
    kz1, Pk_z1 = compute_pk_z(data1, Lz)
    kz2, Pk_z2 = compute_pk_z(data2, Lz)

    # Compute 3D P(k) if requested
    if args.full_3d:
        print("Computing 3D P(k)...")
        k3d_1, Pk3d_1, _ = compute_pk_3d(data1, Lx, Ly, Lz)
        k3d_2, Pk3d_2, _ = compute_pk_3d(data2, Lx, Ly, Lz)

    # Create figure
    if args.full_3d:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- P(kz) comparison ---
    ax = axes[0, 0]
    ax.loglog(kz1[1:], Pk_z1[1:], 'b-', lw=1.5, label=label1, alpha=0.8)
    ax.loglog(kz2[1:], Pk_z2[1:], 'r--', lw=1.5, label=label2, alpha=0.8)

    # Mark reference scales
    k_fund = 2 * np.pi / Lz
    ax.axvline(k_fund, color='gray', ls=':', alpha=0.5, label=f'k_fund={k_fund:.4f}')

    if args.cube_size is not None:
        k_cube = 2 * np.pi / args.cube_size
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7,
                   label=f'k_cube={k_cube:.4f}')
    elif args.n_cubes is not None:
        L_cube = Lz / args.n_cubes
        k_cube = 2 * np.pi / L_cube
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7,
                   label=f'k_cube ({args.n_cubes} cubes)={k_cube:.4f}')

    ax.set_xlabel('kz [h/Mpc]')
    ax.set_ylabel('P(kz) [Mpc/h]')
    ax.set_title('P(kz) Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Ratio P2/P1 ---
    ax = axes[0, 1]

    # Interpolate to common k values if needed
    if len(kz1) == len(kz2) and np.allclose(kz1, kz2):
        kz_common = kz1[1:]
        ratio = Pk_z2[1:] / Pk_z1[1:]
    else:
        kz_common = kz1[1:]
        Pk_z2_interp = np.interp(kz_common, kz2[1:], Pk_z2[1:])
        ratio = Pk_z2_interp / Pk_z1[1:]

    ax.semilogx(kz_common, ratio, 'g-', lw=1.5)
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.fill_between(kz_common, 0.99, 1.01, color='gray', alpha=0.2, label='1% band')

    if args.cube_size is not None:
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7)
    elif args.n_cubes is not None:
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7)

    ax.set_xlabel('kz [h/Mpc]')
    ax.set_ylabel(f'P(kz) ratio: {label2} / {label1}')
    ax.set_title('Power Spectrum Ratio')
    ax.set_ylim(0.9, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Difference (P2 - P1) / P1 ---
    ax = axes[1, 0]
    frac_diff = (Pk_z2[1:] - Pk_z1[1:]) / Pk_z1[1:] * 100  # percentage

    ax.semilogx(kz1[1:], frac_diff, 'm-', lw=1.5)
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.fill_between(kz1[1:], -1, 1, color='gray', alpha=0.2, label='1% band')

    if args.cube_size is not None:
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7)
    elif args.n_cubes is not None:
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7)

    ax.set_xlabel('kz [h/Mpc]')
    ax.set_ylabel('Fractional difference [%]')
    ax.set_title(f'({label2} - {label1}) / {label1} x 100%')
    ax.set_ylim(-10, 10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Difference in linear scale ---
    ax = axes[1, 1]
    abs_diff = Pk_z2[1:] - Pk_z1[1:]

    ax.semilogx(kz1[1:], abs_diff, 'c-', lw=1.5)
    ax.axhline(0, color='k', ls='--', alpha=0.5)

    if args.cube_size is not None:
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7)
    elif args.n_cubes is not None:
        ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7)

    ax.set_xlabel('kz [h/Mpc]')
    ax.set_ylabel('P(kz) difference [Mpc/h]')
    ax.set_title(f'{label2} - {label1}')
    ax.grid(True, alpha=0.3)

    # --- 3D P(k) if requested ---
    if args.full_3d:
        ax = axes[0, 2]
        ax.loglog(k3d_1, Pk3d_1, 'b-', lw=1.5, label=label1, alpha=0.8)
        ax.loglog(k3d_2, Pk3d_2, 'r--', lw=1.5, label=label2, alpha=0.8)
        ax.set_xlabel('k [h/Mpc]')
        ax.set_ylabel('P(k) [(Mpc/h)³]')
        ax.set_title('3D P(k) Comparison')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        # Interpolate for ratio
        Pk3d_2_interp = np.interp(k3d_1, k3d_2, Pk3d_2)
        ratio_3d = Pk3d_2_interp / Pk3d_1
        ax.semilogx(k3d_1, ratio_3d, 'g-', lw=1.5)
        ax.axhline(1.0, color='k', ls='--', alpha=0.5)
        ax.fill_between(k3d_1, 0.99, 1.01, color='gray', alpha=0.2)
        ax.set_xlabel('k [h/Mpc]')
        ax.set_ylabel('3D P(k) ratio')
        ax.set_title('3D Power Spectrum Ratio')
        ax.set_ylim(0.9, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Power Spectrum Comparison\n{label1} vs {label2}', fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"P(kz) ratio statistics (file2/file1):")
    print(f"  Mean ratio:  {ratio.mean():.6f}")
    print(f"  Std ratio:   {ratio.std():.6f}")
    print(f"  Min ratio:   {ratio.min():.6f}")
    print(f"  Max ratio:   {ratio.max():.6f}")
    print(f"\nFractional difference:")
    print(f"  Mean: {frac_diff.mean():.3f}%")
    print(f"  Std:  {frac_diff.std():.3f}%")
    print(f"  Max:  {np.abs(frac_diff).max():.3f}%")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    exit(main())
