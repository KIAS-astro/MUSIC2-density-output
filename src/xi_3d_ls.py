#!/usr/bin/env python3
"""
Compute the 3-D isotropic two-point correlation function xi(r) using the
Landy-Szalay estimator for three MUSIC2 delta fields (FS, S1, S2).

Follows ppp2.c logic with fully vectorised pair counting:

    den     = 1 + delta          (data field)
    den_ran = 1.0                (uniform grid "random", same positions)

For each lag vector (di, dj, dk) with physical distance
    r = sqrt((di*dx)^2 + (dj*dy)^2 + (dk*dz)^2)

the pair sums are accumulated as:
    DD[b] += 2 * sum(den[lo] * den[hi])
    DR[b] += 2 * sum(den[lo])           (den_ran = 1)
    RR[b] += 2 * n_pairs                (1 * 1 = 1)

where [lo] and [hi] are the non-overlapping sub-arrays at that lag,
and the factor 2 accounts for the symmetric (hi,lo) direction.

NO periodic boundary conditions: each lag (di,dj,dk) uses only the
(nx-di)*(ny-dj)*(nz-dk) valid non-wrap pairs.

The grid is assumed cubic-cell if --Lx/Ly are not given (dx=dy=dz=Lz/nz).
For non-cubic boxes supply --Lx and --Ly explicitly.

Optionally crop FS via --ref-left/--ref-right (same as xi_z.py / xi_z_ls.py).

Usage:
    python xi_3d_ls.py --fullsize fullsize_cropped.dat \\
                       --single1  single/delta_level8_real.dat \\
                       --single2  merge/cube0/delta_level8_real.dat \\
                       --Lz 20 --nbins 30 --plot-output xi_3d_ls.png

    # Non-cubic box
    python xi_3d_ls.py ... --Lz 20 --Lx 10 --Ly 10

    # Crop FS on-the-fly
    python xi_3d_ls.py ... --Lz 20 \\
        --ref-left 0.375,0.375,0 --ref-right 0.625,0.625,1

Performance note
----------------
The outer loop is over all unique lag vectors (di, dj, dk).  For a field of
size nx*ny*nz the number of lag vectors is O(nx*ny*nz), but each inner
operation is a fully vectorised numpy dot/sum over the (nx-di)*(ny-dj)*(nz-dk)
sub-array, so no Python-level loops over individual cells.

For large fields (nx,ny ~ 64, nz ~ 256) this can be slow.  Use --dmax to
limit the maximum lag (only lag vectors with r <= dmax are evaluated), which
drastically reduces the number of lag vectors.
"""

import numpy as np
import struct
import argparse
import matplotlib.pyplot as plt
from itertools import product


# ── I/O ──────────────────────────────────────────────────────────────────────

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


def crop_field(data, size, ref_left, ref_right):
    nx, ny, nz = size
    x0 = int(ref_left[0]  * nx);  x1 = int(ref_right[0] * nx)
    y0 = int(ref_left[1]  * ny);  y1 = int(ref_right[1] * ny)
    z0 = int(ref_left[2]  * nz);  z1 = int(ref_right[2] * nz)
    return data[x0:x1, y0:y1, z0:z1].copy(), (x0, x1, y0, y1, z0, z1)


# ── LS estimator ─────────────────────────────────────────────────────────────

def xi_3d_ls(field, Lx, Ly, Lz, bin_edges, dmax=None, verbose=True):
    """3-D isotropic Landy-Szalay xi(r), fully vectorised, non-periodic.

    Parameters
    ----------
    field     : ndarray (nx, ny, nz)
    Lx,Ly,Lz : float  physical box lengths [Mpc]
    bin_edges : 1-D array [Mpc]
    dmax      : float or None  maximum lag to consider [Mpc].
                If None uses bin_edges[-1].  Setting this limits which
                lag vectors (di,dj,dk) are iterated, saving time.
    verbose   : bool  print progress

    Returns
    -------
    xi   : 1-D array (nbins,)
    r_c  : 1-D array (nbins,)  bin centres [Mpc]
    DD, DR, RR : raw accumulated counts
    """
    nx, ny, nz = field.shape
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    nbins = len(bin_edges) - 1
    r_c   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if dmax is None:
        dmax = bin_edges[-1]

    # Maximum integer lags in each direction
    di_max = min(nx - 1, int(np.ceil(dmax / dx)))
    dj_max = min(ny - 1, int(np.ceil(dmax / dy)))
    dk_max = min(nz - 1, int(np.ceil(dmax / dz)))

    DD = np.zeros(nbins)
    DR = np.zeros(nbins)
    RR = np.zeros(nbins)

    den = 1.0 + field   # data weights (nx, ny, nz)

    # Enumerate unique lag vectors in the positive half-space:
    #   dk > 0, OR (dk==0 AND dj > 0), OR (dk==0 AND dj==0 AND di > 0)
    # Each contributes factor 2 (symmetric partner).
    n_lags = 0
    total_lags = (di_max + 1) * (dj_max + 1) * (dk_max + 1) - 1

    for dk in range(0, dk_max + 1):
        dj_range = range(0, dj_max + 1) if dk > 0 else range(0, dj_max + 1)
        for dj in dj_range:
            di_start = 1 if (dk == 0 and dj == 0) else 0
            di_range = range(di_start, di_max + 1)
            for di in di_range:
                # Skip (0,0,0)
                if dk == 0 and dj == 0 and di == 0:
                    continue

                r = np.sqrt((di*dx)**2 + (dj*dy)**2 + (dk*dz)**2)
                if r > dmax:
                    continue

                b = np.searchsorted(bin_edges, r, side='right') - 1
                if not (0 <= b < nbins):
                    continue

                # Sub-arrays: lo = lower corner, hi = shifted corner
                lo = den[:nx-di if di > 0 else nx,
                         :ny-dj if dj > 0 else ny,
                         :nz-dk if dk > 0 else nz]
                hi = den[di:, dj:, dk:]

                n_pairs = lo.size   # (nx-di)*(ny-dj)*(nz-dk)

                # Factor 2: count (lo,hi) and symmetric (hi,lo)
                DD[b] += 2.0 * np.dot(lo.ravel(), hi.ravel())
                DR[b] += 2.0 * lo.sum()
                RR[b] += 2.0 * n_pairs

                n_lags += 1

        if verbose and dk % max(1, dk_max//10) == 0:
            print(f"  lag dk={dk}/{dk_max} ...", flush=True)

    if verbose:
        print(f"  Total lag vectors evaluated: {n_lags}")

    with np.errstate(invalid='ignore', divide='ignore'):
        xi = np.where(RR > 0, (DD - 2.0*DR + RR) / RR, np.nan)

    return xi, r_c, DD, DR, RR


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-D Landy-Szalay xi(r) for FS/S1/S2 (vectorised, non-periodic)")
    parser.add_argument("--fullsize", required=True)
    parser.add_argument("--single1",  required=True)
    parser.add_argument("--single2",  required=True)
    parser.add_argument("--Lz", type=float, required=True,
                        help="Physical z-length [Mpc]")
    parser.add_argument("--Lx", type=float, default=None,
                        help="Physical x-length [Mpc] (default: same as Lz)")
    parser.add_argument("--Ly", type=float, default=None,
                        help="Physical y-length [Mpc] (default: same as Lz)")
    # crop
    parser.add_argument("--ref-left",  default=None,
                        help="Fractional lower corner for FS crop, e.g. 0.375,0.375,0")
    parser.add_argument("--ref-right", default=None,
                        help="Fractional upper corner for FS crop, e.g. 0.625,0.625,1")
    # binning
    parser.add_argument("--nbins", type=int, default=30,
                        help="Number of r bins (default: 30)")
    parser.add_argument("--rmin", type=float, default=None,
                        help="Minimum lag [Mpc] (default: cell diagonal / sqrt(3))")
    parser.add_argument("--rmax", type=float, default=None,
                        help="Maximum lag [Mpc] (default: min(Lx,Ly,Lz)/2)")
    parser.add_argument("--dmax", type=float, default=None,
                        help="Maximum lag for iteration [Mpc] (default: rmax). "
                             "Set equal to rmax to avoid evaluating lags outside "
                             "the plot range (speeds up computation).")
    # plot
    parser.add_argument("--labels", default="FS,S1,S2")
    parser.add_argument("--r2",   action="store_true",
                        help="Plot xi(r)*r^2")
    parser.add_argument("--logy", action="store_true",
                        help="Log y-axis (|xi|)")
    parser.add_argument("--plot-output", default="xi_3d_ls.png")
    args = parser.parse_args()

    if (args.ref_left is None) != (args.ref_right is None):
        parser.error("--ref-left and --ref-right must be given together")
    do_crop = args.ref_left is not None
    if do_crop:
        ref_left  = [float(x) for x in args.ref_left.split(',')]
        ref_right = [float(x) for x in args.ref_right.split(',')]

    labels = [s.strip() for s in args.labels.split(',')]
    if len(labels) != 3:
        parser.error("--labels must have 3 entries")

    # ── Read ─────────────────────────────────────────────────────────────────
    print("Reading fields...")
    fs_data, fs_size, _ = read_field(args.fullsize)
    s1_data, s1_size, _ = read_field(args.single1)
    s2_data, s2_size, _ = read_field(args.single2)
    print(f"  {labels[0]} (raw): {fs_size[0]}x{fs_size[1]}x{fs_size[2]}, "
          f"δ ∈ [{fs_data.min():.4e}, {fs_data.max():.4e}]")
    print(f"  {labels[1]}: {s1_size[0]}x{s1_size[1]}x{s1_size[2]}, "
          f"δ ∈ [{s1_data.min():.4e}, {s1_data.max():.4e}]")
    print(f"  {labels[2]}: {s2_size[0]}x{s2_size[1]}x{s2_size[2]}, "
          f"δ ∈ [{s2_data.min():.4e}, {s2_data.max():.4e}]")

    if do_crop:
        fs_data, cidx = crop_field(fs_data, fs_size, ref_left, ref_right)
        x0,x1,y0,y1,z0,z1 = cidx
        print(f"\n  Cropped {labels[0]} [{x0}:{x1},{y0}:{y1},{z0}:{z1}] "
              f"→ {fs_data.shape}")
        if fs_data.shape != s1_data.shape:
            print(f"  WARNING: FS {fs_data.shape} != S1 {s1_data.shape}")
        if fs_data.shape != s2_data.shape:
            print(f"  WARNING: FS {fs_data.shape} != S2 {s2_data.shape}")
    else:
        print(f"\n  No crop applied to {labels[0]}")

    # ── Box sizes ─────────────────────────────────────────────────────────────
    nx, ny, nz = fs_data.shape
    Lz = args.Lz
    Lx = args.Lx if args.Lx is not None else Lz * nx / nz
    Ly = args.Ly if args.Ly is not None else Lz * ny / nz
    dx = Lx / nx;  dy = Ly / ny;  dz = Lz / nz
    print(f"\n  Box: Lx={Lx:.3f} Ly={Ly:.3f} Lz={Lz:.3f} Mpc")
    print(f"  Cell: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} Mpc")

    # ── Bins ─────────────────────────────────────────────────────────────────
    rmin = args.rmin if args.rmin is not None else np.sqrt(dx**2+dy**2+dz**2)
    rmax = args.rmax if args.rmax is not None else min(Lx, Ly, Lz) / 2.0
    dmax = args.dmax if args.dmax is not None else rmax
    bin_edges = np.linspace(rmin, rmax, args.nbins + 1)
    print(f"  Bins: {args.nbins} in [{rmin:.3f}, {rmax:.3f}] Mpc,  dmax={dmax:.3f} Mpc")

    # ── Compute ──────────────────────────────────────────────────────────────
    fields = [fs_data, s1_data, s2_data]
    xis, r_cs = [], []

    for data, name in zip(fields, labels):
        print(f"\nComputing 3-D LS xi for {name}  (shape={data.shape}) ...")
        xi, r_c, DD, DR, RR = xi_3d_ls(
            data, Lx, Ly, Lz, bin_edges, dmax=dmax, verbose=True)
        xis.append(xi)
        r_cs.append(r_c)
        finite = np.isfinite(xi)
        print(f"  xi ∈ [{np.nanmin(xi):.4e}, {np.nanmax(xi):.4e}]  "
              f"({finite.sum()}/{len(xi)} finite bins)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors     = ['C0', 'C1', 'C2']
    linestyles = ['-',  '--', '-.']

    fig, ax = plt.subplots(figsize=(8, 5))

    for xi, r_c, name, color, ls in zip(xis, r_cs, labels, colors, linestyles):
        v   = np.isfinite(xi)
        r_v = r_c[v]
        y   = xi[v] * r_v**2 if args.r2 else xi[v]
        y   = np.abs(y)       if args.logy else y
        ax.plot(r_v, y, color=color, ls=ls, lw=1.8, label=name)

    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'$r$  [Mpc]', fontsize=13)

    if args.r2:
        ylabel = r'$\xi_{\rm LS}(r)\,r^2$  [Mpc$^2$]'
    else:
        ylabel = r'$\xi_{\rm LS}(r)$'
    if args.logy:
        ax.set_yscale('log')
        ylabel = (r'$|\xi_{\rm LS}|\,r^2$' if args.r2
                  else r'$|\xi_{\rm LS}(r)|$')
    ax.set_ylabel(ylabel, fontsize=13)

    crop_note = (f'FS cropped [{args.ref_left}]–[{args.ref_right}]'
                 if do_crop else 'no FS crop')
    ax.set_title(
        f'3-D Landy–Szalay $\\xi(r)$ — ppp2.c style, non-periodic\n'
        f'$L_z={Lz}$ Mpc,  {nx}×{ny}×{nz},  '
        f'$\\Delta z={dz:.3f}$ Mpc/cell  |  {crop_note}',
        fontsize=10)
    ax.legend(fontsize=11)
    ax.set_xlim(rmin, rmax)
    fig.tight_layout()
    fig.savefig(args.plot_output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.plot_output}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    exit(main())
