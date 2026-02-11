#!/usr/bin/env python3
"""
Calculate power spectrum along z direction from a MUSIC2 .dat field.

This computes P(kz) by FFT along z and averaging over all (x,y) positions.
Useful for detecting artifacts at specific z-scales (e.g., cube boundaries).

Usage:
    python power_spectrum_z.py field.dat --Lz 100.0
    python power_spectrum_z.py field.dat --Lx 25 --Ly 25 --Lz 100
    python power_spectrum_z.py field.dat --Lz 100 --plot
    python power_spectrum_z.py field.dat --Lz 100 --full-3d
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


def compute_pk_z(data, Lz):
    """Compute 1D power spectrum along z direction.

    For each (x,y), FFT along z, then average |delta(kz)|^2 over all (x,y).

    Returns:
        kz: wavenumbers [h/Mpc or 1/Mpc depending on Lz units]
        Pk: power spectrum P(kz)
    """
    nx, ny, nz = data.shape

    # FFT along z axis for all (x,y)
    delta_kz = np.fft.rfft(data, axis=2)  # shape: (nx, ny, nz//2+1)

    # Power = |delta_k|^2, averaged over (x,y)
    power = np.abs(delta_kz) ** 2
    Pk = power.mean(axis=(0, 1))  # average over x, y

    # Normalization: divide by nz^2 for FFT normalization
    Pk /= nz ** 2

    # Multiply by Lz to get proper P(k) units [Mpc/h or Mpc]
    Pk *= Lz

    # Wavenumbers
    kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi  # [h/Mpc or 1/Mpc]

    return kz, Pk


def compute_pk_3d(data, Lx, Ly, Lz):
    """Compute full 3D power spectrum P(k), binned by |k|.

    Returns:
        k: wavenumber bins
        Pk: power spectrum P(k)
        Nmodes: number of modes per bin
    """
    nx, ny, nz = data.shape

    # 3D FFT
    delta_k = np.fft.rfftn(data)

    # Wavenumber arrays
    kx = np.fft.fftfreq(nx, d=Lx / nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=Ly / ny) * 2 * np.pi
    kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi

    # 3D k magnitude grid
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    # Power
    power = np.abs(delta_k) ** 2

    # Normalization
    power /= (nx * ny * nz) ** 2
    V = Lx * Ly * Lz
    power *= V

    # Bin by |k|
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

    # Filter out empty bins
    valid = Nmodes > 0
    return k_centers[valid], Pk[valid], Nmodes[valid]


def main():
    parser = argparse.ArgumentParser(
        description="Calculate power spectrum along z direction.")
    parser.add_argument("file", help="Path to the .dat file")
    parser.add_argument("--Lx", type=float, default=None,
                        help="Box length in x [Mpc/h] (default: same as Lz)")
    parser.add_argument("--Ly", type=float, default=None,
                        help="Box length in y [Mpc/h] (default: same as Lz)")
    parser.add_argument("--Lz", type=float, required=True,
                        help="Box length in z [Mpc/h]")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the power spectrum")
    parser.add_argument("--full-3d", action="store_true", dest="full_3d",
                        help="Also compute full 3D P(k)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output text file for P(k) data")
    parser.add_argument("--plot-output", default=None,
                        help="Output plot filename (default: pk_z.png)")
    parser.add_argument("--cube-size", type=float, default=None, dest="cube_size",
                        help="Size of each merged cube in z [Mpc/h], to mark artifact scale")
    parser.add_argument("--n-cubes", type=int, default=None, dest="n_cubes",
                        help="Number of cubes merged, to mark artifact scale")
    args = parser.parse_args()

    # Set default Lx, Ly
    Lx = args.Lx if args.Lx is not None else args.Lz
    Ly = args.Ly if args.Ly is not None else args.Lz
    Lz = args.Lz

    # Read data
    print(f"Reading {args.file}...")
    data, size, offset = read_field(args.file)
    nx, ny, nz = size
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    print(f"Box dimensions: Lx={Lx}, Ly={Ly}, Lz={Lz} Mpc/h")
    print(f"Cell size: dx={Lx/nx:.4f}, dy={Ly/ny:.4f}, dz={Lz/nz:.4f} Mpc/h")
    print(f"Data mean: {data.mean():.6e}, std: {data.std():.6e}")

    # Compute P(kz)
    print("\nComputing P(kz)...")
    kz, Pk_z = compute_pk_z(data, Lz)

    print(f"\nP(kz) results:")
    print(f"  k_fundamental (z): {2*np.pi/Lz:.6f} h/Mpc")
    print(f"  k_Nyquist (z):     {np.pi*nz/Lz:.6f} h/Mpc")
    print(f"  Number of modes:   {len(kz)}")

    # Print table
    print(f"\n{'kz [h/Mpc]':>14s}  {'P(kz)':>14s}")
    print("-" * 32)
    for i in range(min(20, len(kz))):
        print(f"{kz[i]:14.6f}  {Pk_z[i]:14.6e}")
    if len(kz) > 20:
        print(f"  ... ({len(kz) - 20} more rows)")

    # Compute full 3D P(k) if requested
    if args.full_3d:
        print("\nComputing full 3D P(k)...")
        k_3d, Pk_3d, Nmodes = compute_pk_3d(data, Lx, Ly, Lz)
        print(f"  Number of k bins: {len(k_3d)}")

    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"# Power spectrum from {args.file}\n")
            f.write(f"# Grid: {nx} x {ny} x {nz}\n")
            f.write(f"# Box: Lx={Lx}, Ly={Ly}, Lz={Lz} Mpc/h\n")
            f.write(f"# kz [h/Mpc]    P(kz)\n")
            for i in range(len(kz)):
                f.write(f"{kz[i]:.8e}  {Pk_z[i]:.8e}\n")
        print(f"\nSaved P(kz) to {args.output}")

    # Plot
    if args.plot:
        import matplotlib.pyplot as plt

        if args.full_3d:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(7, 5))
            axes = [axes]

        # P(kz) plot
        ax = axes[0]
        ax.loglog(kz[1:], Pk_z[1:], 'b-', lw=1.5, label='P(kz)')
        ax.set_xlabel('kz [h/Mpc]')
        ax.set_ylabel('P(kz) [(Mpc/h)]')
        ax.set_title('1D Power Spectrum along z')
        ax.grid(True, alpha=0.3)

        # Mark fundamental mode
        k_fund_z = 2 * np.pi / Lz
        ax.axvline(k_fund_z, color='r', ls='--', alpha=0.5,
                   label=f'k_fundamental = {k_fund_z:.4f} h/Mpc')

        # Mark cube artifact scale if specified
        if args.cube_size is not None:
            k_cube = 2 * np.pi / args.cube_size
            ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7,
                       label=f'k_cube (L={args.cube_size}) = {k_cube:.4f} h/Mpc')
            # Also mark harmonics
            for n in [2, 3]:
                k_harm = n * k_cube
                if k_harm < kz.max():
                    ax.axvline(k_harm, color='orange', ls=':', alpha=0.5,
                               label=f'  {n}x harmonic = {k_harm:.4f}')
        elif args.n_cubes is not None:
            L_cube = Lz / args.n_cubes
            k_cube = 2 * np.pi / L_cube
            ax.axvline(k_cube, color='orange', ls='-', lw=2, alpha=0.7,
                       label=f'k_cube ({args.n_cubes} cubes) = {k_cube:.4f} h/Mpc')

        ax.legend(fontsize=9, loc='best')

        # Full 3D P(k) plot
        if args.full_3d:
            ax = axes[1]
            ax.loglog(k_3d, Pk_3d, 'g-', lw=1.5, label='P(k) 3D')
            ax.set_xlabel('k [h/Mpc]')
            ax.set_ylabel('P(k) [(Mpc/h)³]')
            ax.set_title('3D Power Spectrum')
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.suptitle(f'{args.file}\nGrid: {nx}x{ny}x{nz}, '
                     f'Box: {Lx}x{Ly}x{Lz} Mpc/h', fontsize=10)
        fig.tight_layout()

        outfile = args.plot_output if args.plot_output else "pk_z.png"
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {outfile}")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    exit(main())
