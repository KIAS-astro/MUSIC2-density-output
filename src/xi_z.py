#!/usr/bin/env python3
"""
Compute the 1-D two-point correlation function along the z-axis for three
MUSIC2 delta fields (FS, S1, S2) and overplot them.

The estimator averages the z-direction autocorrelation over all (x, y) pencils:

    xi(r) = < delta(z) * delta(z + r) >_{x,y,z}  /  <delta^2>_{x,y,z}

which is equivalent to:
    1. For each (x,y) pencil compute the autocorrelation via FFT.
    2. Average over all pencils.
    3. Normalise by the zero-lag value so xi(0) = 1  (optional, see --no-norm).

The lag r is converted to physical Mpc using --Lz (total box length in z).

When --ref-left / --ref-right are supplied, the FS field is cropped to the
same spatial sub-region as S1/S2 before computing xi, so that all three curves
use exactly the same pencils.  The --Lz argument should then be the physical
z-extent of the cropped region (i.e. ref_z_fraction * full_Lz).

Usage:
    # No crop (pass already-cropped FS file, or compare full fields)
    python xi_z.py --fullsize fullsize_cropped.dat \
                   --single1  single/delta_level8_real.dat \
                   --single2  merge/cube0/delta_level8_real.dat \
                   --Lz 200 --plot-output xi_z.png

    # Crop FS on-the-fly to the zoom-in region before computing xi
    python xi_z.py --fullsize fullsize/delta_level8_real.dat \
                   --single1  single/delta_level8_real.dat \
                   --single2  merge/cube0/delta_level8_real.dat \
                   --Lz 200 \
                   --ref-left 0.375,0.375,0 --ref-right 0.625,0.625,1 \
                   --plot-output xi_z.png

    # Override field labels / restrict plot range
    python xi_z.py ... --labels "FullBox,Single,Merge" --rmax 50
"""

import numpy as np
import struct
import argparse
import matplotlib.pyplot as plt


# ── I/O ──────────────────────────────────────────────────────────────────────

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


def crop_field(data, size, ref_left, ref_right):
    """Crop data to the refinement sub-region.

    Parameters
    ----------
    data     : ndarray (nx, ny, nz)
    size     : tuple (nx, ny, nz)
    ref_left : list [fx, fy, fz]  fractional coordinates of lower corner
    ref_right: list [fx, fy, fz]  fractional coordinates of upper corner

    Returns
    -------
    cropped  : ndarray sub-array
    crop_idx : tuple (x_start, x_end, y_start, y_end, z_start, z_end)
    """
    nx, ny, nz = size
    x_start = int(ref_left[0]  * nx);  x_end = int(ref_right[0] * nx)
    y_start = int(ref_left[1]  * ny);  y_end = int(ref_right[1] * ny)
    z_start = int(ref_left[2]  * nz);  z_end = int(ref_right[2] * nz)
    cropped = data[x_start:x_end, y_start:y_end, z_start:z_end].copy()
    return cropped, (x_start, x_end, y_start, y_end, z_start, z_end)


# ── Correlation ───────────────────────────────────────────────────────────────

def xi_along_z(field):
    """Compute the mean 1-D autocorrelation function along the z-axis.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz)

    Returns
    -------
    xi : ndarray, shape (nz,)
        Mean autocorrelation averaged over all (x, y) pencils.
        NOT normalised here; normalisation is done by the caller.
    lags : ndarray, shape (nz,)
        Integer lag indices 0 .. nz-1.
    """
    nx, ny, nz = field.shape

    # FFT-based autocorrelation for every (x,y) pencil at once.
    F   = np.fft.rfft(field, axis=2)                        # (nx, ny, nz//2+1)
    acf = np.fft.irfft(F * np.conj(F), n=nz, axis=2)       # (nx, ny, nz)

    # Average over all pencils, normalise by nz (irfft convention)
    xi_mean = acf.mean(axis=(0, 1)) / nz                    # (nz,)

    return xi_mean, np.arange(nz)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="1-D two-point correlation function along z for FS/S1/S2 delta fields")
    parser.add_argument("--fullsize", required=True,
                        help="Path to fullsize (or pre-cropped) delta field")
    parser.add_argument("--single1",  required=True,
                        help="Path to first single delta field")
    parser.add_argument("--single2",  required=True,
                        help="Path to second single delta field")
    parser.add_argument("--Lz", type=float, required=True,
                        help="Physical z-length used for the lag axis [Mpc]. "
                             "If --ref-left/right are given, this should be the "
                             "z-extent of the cropped region "
                             "(ref_z_fraction * full_Lz).")
    # ── crop options ────────────────────────────────────────────────────────
    parser.add_argument("--ref-left", default=None,
                        help="Fractional lower corner for FS crop, e.g. "
                             "0.375,0.375,0  (x,y,z).  "
                             "When set, FS is cropped before computing xi so "
                             "it covers the same region as S1/S2.")
    parser.add_argument("--ref-right", default=None,
                        help="Fractional upper corner for FS crop, e.g. "
                             "0.625,0.625,1  (x,y,z).")
    # ── other options ────────────────────────────────────────────────────────
    parser.add_argument("--labels", default="FS,S1,S2",
                        help="Comma-separated curve labels (default: FS,S1,S2)")
    parser.add_argument("--rmax", type=float, default=None,
                        help="Maximum lag to plot [Mpc] (default: Lz/2)")
    parser.add_argument("--no-norm", action="store_true",
                        help="Do not normalise xi so that xi(0)=1")
    parser.add_argument("--plot-output", default="xi_z.png",
                        help="Output plot filename (default: xi_z.png)")
    parser.add_argument("--logy", action="store_true",
                        help="Use log scale on y-axis (absolute value)")
    parser.add_argument("--r2", action="store_true",
                        help="Multiply xi by r^2, i.e. plot xi(r)*r^2  "
                             "(r=0 point is dropped to avoid xi(0)*0^2 spike)")
    args = parser.parse_args()

    # Validate crop arguments: must supply both or neither
    if (args.ref_left is None) != (args.ref_right is None):
        parser.error("--ref-left and --ref-right must be supplied together")
    do_crop = args.ref_left is not None

    if do_crop:
        ref_left  = [float(x) for x in args.ref_left.split(',')]
        ref_right = [float(x) for x in args.ref_right.split(',')]
        if len(ref_left) != 3 or len(ref_right) != 3:
            parser.error("--ref-left/right must each have exactly 3 comma-separated values")

    labels = [s.strip() for s in args.labels.split(',')]
    if len(labels) != 3:
        parser.error("--labels must have exactly 3 comma-separated entries")

    # ── Read fields ──────────────────────────────────────────────────────────
    print("Reading fields...")
    fs_data,  fs_size,  fs_offset  = read_field(args.fullsize)
    s1_data,  s1_size,  s1_offset  = read_field(args.single1)
    s2_data,  s2_size,  s2_offset  = read_field(args.single2)

    print(f"  {labels[0]} (raw): {fs_size[0]}x{fs_size[1]}x{fs_size[2]}, "
          f"offset={fs_offset}, δ ∈ [{fs_data.min():.4e}, {fs_data.max():.4e}]")
    print(f"  {labels[1]}: {s1_size[0]}x{s1_size[1]}x{s1_size[2]}, "
          f"offset={s1_offset}, δ ∈ [{s1_data.min():.4e}, {s1_data.max():.4e}]")
    print(f"  {labels[2]}: {s2_size[0]}x{s2_size[1]}x{s2_size[2]}, "
          f"offset={s2_offset}, δ ∈ [{s2_data.min():.4e}, {s2_data.max():.4e}]")

    # ── Optionally crop FS ───────────────────────────────────────────────────
    if do_crop:
        fs_data, crop_idx = crop_field(fs_data, fs_size, ref_left, ref_right)
        x0, x1, y0, y1, z0, z1 = crop_idx
        print(f"\n  Cropped {labels[0]} to [{x0}:{x1}, {y0}:{y1}, {z0}:{z1}]  "
              f"→  {fs_data.shape[0]}x{fs_data.shape[1]}x{fs_data.shape[2]}")
        # Warn if cropped FS doesn't match S1/S2
        if fs_data.shape != s1_data.shape:
            print(f"  WARNING: cropped FS shape {fs_data.shape} != S1 {s1_data.shape}")
        if fs_data.shape != s2_data.shape:
            print(f"  WARNING: cropped FS shape {fs_data.shape} != S2 {s2_data.shape}")
    else:
        print(f"\n  No crop applied to {labels[0]} (using full array as-is)")

    fields = [fs_data, s1_data, s2_data]

    # Check all fields have the same nz
    nz_list = [f.shape[2] for f in fields]
    if len(set(nz_list)) != 1:
        print(f"\nWARNING: fields have different nz: {nz_list}")
        print("  Using nz of FS for the lag axis.")

    nz       = fields[0].shape[2]
    dz       = args.Lz / nz
    lags_mpc = np.arange(nz) * dz

    # ── Compute xi ───────────────────────────────────────────────────────────
    print("\nComputing xi(r) along z ...")
    xis = []
    for data, name in zip(fields, labels):
        xi, _ = xi_along_z(data)

        if not args.no_norm:
            xi0 = xi[0]
            if xi0 == 0:
                print(f"  WARNING: xi(0)=0 for {name}, skipping normalisation")
            else:
                xi /= xi0

        xis.append(xi)
        print(f"  {name}: shape={data.shape}, n_pencils={data.shape[0]*data.shape[1]}, "
              f"xi(0)={xi[0]:.6e}, xi(1*dz={dz:.3f} Mpc)={xi[1]:.6e}")

    # ── Determine plot range ─────────────────────────────────────────────────
    rmax = args.rmax if args.rmax is not None else args.Lz / 2
    mask = lags_mpc <= rmax

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors     = ['C0', 'C1', 'C2']
    linestyles = ['-',  '--', '-.']

    fig, ax = plt.subplots(figsize=(8, 5))

    r_plot = lags_mpc[mask]
    if args.r2:
        skip    = (r_plot > 0)
        r_plot  = r_plot[skip]
        xis_plt = [xi[mask][skip] for xi in xis]
    else:
        xis_plt = [xi[mask] for xi in xis]

    for xi_plt, name, color, ls in zip(xis_plt, labels, colors, linestyles):
        y = xi_plt * r_plot**2 if args.r2 else xi_plt
        y = np.abs(y)          if args.logy else y
        ax.plot(r_plot, y, color=color, ls=ls, lw=1.8, label=name)

    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'$r_z$  [Mpc]', fontsize=13)

    if args.r2:
        ylabel = (r'$\xi(r_z)\,r_z^2$' if args.no_norm
                  else r'$\xi(r_z)\,r_z^2\,/\,\xi(0)$  [Mpc$^2$]')
    else:
        ylabel = (r'$\xi(r_z)$' if args.no_norm
                  else r'$\xi(r_z)\,/\,\xi(0)$')
    ax.set_ylabel(ylabel, fontsize=13)

    if args.logy:
        ax.set_yscale('log')
        ax.set_ylabel((r'$|\xi(r_z)|\,r_z^2$' if args.r2
                       else r'$|\xi(r_z)|$'), fontsize=13)

    crop_note = (f'FS cropped to [{args.ref_left}]–[{args.ref_right}]'
                 if do_crop else 'no FS crop')
    ax.set_title(
        f'1-D two-point correlation function along z\n'
        f'$L_z={args.Lz}$ Mpc,  $n_z={nz}$,  '
        f'$\\Delta z={dz:.3f}$ Mpc/cell  |  {crop_note}',
        fontsize=10)
    ax.legend(fontsize=11)
    ax.set_xlim(0, rmax)
    fig.tight_layout()
    fig.savefig(args.plot_output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.plot_output}")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    exit(main())
