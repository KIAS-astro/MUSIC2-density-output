#!/usr/bin/env python3
"""
Compute the 1-D two-point correlation function along the z-axis using the
Landy-Szalay estimator for three MUSIC2 delta fields (FS, S1, S2).

This script follows the same logic as ppp2.c:

    den     = 1 + delta          (data field)
    den_ran = 1.0                (uniform "random" field, same grid)

    DD[r] = sum_{pairs at lag r} den[i]     * den[j]
    DR[r] = sum_{pairs at lag r} den[i]     * den_ran[j]   = sum den[i+lag] at each i
    RR[r] = sum_{pairs at lag r} den_ran[i] * den_ran[j]   = n_pairs(r)

    xi_LS(r) = (DD - 2*DR + RR) / RR

Using the uniform grid as the "random catalog" (den_ran=1) rather than a
Monte-Carlo random set eliminates sampling noise entirely and is exactly
equivalent to ppp2.c's approach.

The z-axis is treated as NON-PERIODIC: at lag k*dz only the (nz-k) pairs
(i, i+k) exist, not the wrap-around pairs. This is the boundary correction.

All (x,y) pencils are accumulated before the final LS combination.

Optionally crop FS to the same sub-region as S1/S2 via --ref-left/--ref-right.

Usage:
    python xi_z_ls.py --fullsize fullsize_cropped.dat \\
                      --single1  single/delta_level8_real.dat \\
                      --single2  merge/cube0/delta_level8_real.dat \\
                      --Lz 20 --nbins 30 --plot-output xi_z_ls.png

    # Crop FS on-the-fly
    python xi_z_ls.py --fullsize fullsize/delta_level8_real.dat \\
                      --single1  single/delta_level8_real.dat \\
                      --single2  merge/cube0/delta_level8_real.dat \\
                      --Lz 20 \\
                      --ref-left 0.375,0.375,0 --ref-right 0.625,0.625,1 \\
                      --plot-output xi_z_ls.png
"""

import numpy as np
import struct
import argparse
import matplotlib.pyplot as plt


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
    """Crop data to refinement sub-region (fractional coordinates)."""
    nx, ny, nz = size
    x0 = int(ref_left[0]  * nx);  x1 = int(ref_right[0] * nx)
    y0 = int(ref_left[1]  * ny);  y1 = int(ref_right[1] * ny)
    z0 = int(ref_left[2]  * nz);  z1 = int(ref_right[2] * nz)
    return data[x0:x1, y0:y1, z0:z1].copy(), (x0, x1, y0, y1, z0, z1)


# ── LS estimator ─────────────────────────────────────────────────────────────

def xi_ls_along_z(field, Lz, bin_edges):
    """Landy-Szalay xi(r_z) along z, non-periodic, following ppp2.c logic.

    den     = 1 + delta   (data weights)
    den_ran = 1.0         (uniform grid random weights)

    For each lag k (k = 1 .. nz-1):
        n_pairs = nz - k                    (non-periodic: no wrap-around)
        DD[b] += sum_i den[i] * den[i+k]
        DR[b] += sum_i den[i] * 1           = sum_i den[i] over valid range
        RR[b] += n_pairs                    (= sum_i 1*1)

    Accumulated over all (x,y) pencils, then:
        xi(r) = (DD - 2*DR + RR) / RR

    Parameters
    ----------
    field     : ndarray (nx, ny, nz)
    Lz        : float   physical z-length [Mpc]
    bin_edges : 1-D array [Mpc]

    Returns
    -------
    xi   : 1-D array (nbins,)
    r_c  : 1-D array (nbins,)  bin centres [Mpc]
    DD, DR, RR : raw accumulated counts (for diagnostics)
    """
    nx, ny, nz = field.shape
    dz         = Lz / nz
    nbins      = len(bin_edges) - 1
    r_c        = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Pre-map each lag index to its bin (-1 = outside range)
    lag_bin = np.full(nz, -1, dtype=int)
    for k in range(1, nz):
        r   = k * dz
        idx = np.searchsorted(bin_edges, r, side='right') - 1
        if 0 <= idx < nbins:
            lag_bin[k] = idx

    DD = np.zeros(nbins)
    DR = np.zeros(nbins)
    RR = np.zeros(nbins)

    # den = 1 + delta for every pencil; den_ran = 1 (scalar, no array needed)
    for ix in range(nx):
        for iy in range(ny):
            den = 1.0 + field[ix, iy, :]    # (nz,)

            for k in range(1, nz):
                b = lag_bin[k]
                if b < 0:
                    continue

                n_k    = nz - k              # number of non-periodic pairs
                den_lo = den[:n_k]           # den[i]
                den_hi = den[k:]             # den[i+k]

                # Both directions: (i,j) and (j,i) — symmetric, count once each
                # ppp2.c loops all (i1,i2) so counts each pair twice,
                # but that cancels in (DD-2DR+RR)/RR.  We accumulate once
                # and it cancels identically.
                DD[b] += np.dot(den_lo, den_hi)
                DR[b] += den_lo.sum()        # den[i]*1 summed over valid i
                RR[b] += n_k                 # 1*1 summed over valid pairs

    # Landy-Szalay
    with np.errstate(invalid='ignore', divide='ignore'):
        xi = np.where(RR > 0, (DD - 2.0*DR + RR) / RR, np.nan)

    return xi, r_c, DD, DR, RR


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="1-D Landy-Szalay xi(r_z) for FS/S1/S2 (ppp2.c-style, non-periodic)")
    parser.add_argument("--fullsize", required=True)
    parser.add_argument("--single1",  required=True)
    parser.add_argument("--single2",  required=True)
    parser.add_argument("--Lz", type=float, required=True,
                        help="Physical z-length [Mpc]")
    # crop
    parser.add_argument("--ref-left",  default=None,
                        help="Fractional lower corner for FS crop, e.g. 0.375,0.375,0")
    parser.add_argument("--ref-right", default=None,
                        help="Fractional upper corner for FS crop, e.g. 0.625,0.625,1")
    # binning
    parser.add_argument("--nbins", type=int, default=30,
                        help="Number of r_z bins (default: 30)")
    parser.add_argument("--rmin", type=float, default=None,
                        help="Minimum lag [Mpc] (default: one cell width dz)")
    parser.add_argument("--rmax", type=float, default=None,
                        help="Maximum lag [Mpc] (default: Lz/2)")
    # plot
    parser.add_argument("--labels", default="FS,S1,S2")
    parser.add_argument("--r2",   action="store_true",
                        help="Plot xi(r)*r^2")
    parser.add_argument("--logy", action="store_true",
                        help="Log y-axis (|xi|)")
    parser.add_argument("--plot-output", default="xi_z_ls.png")
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

    # ── Bins ─────────────────────────────────────────────────────────────────
    nz   = fs_data.shape[2]
    dz   = args.Lz / nz
    rmin = args.rmin if args.rmin is not None else dz
    rmax = args.rmax if args.rmax is not None else args.Lz / 2.0
    bin_edges = np.linspace(rmin, rmax, args.nbins + 1)

    print(f"\n  Lz={args.Lz} Mpc,  nz={nz},  dz={dz:.4f} Mpc/cell")
    print(f"  Bins: {args.nbins} in [{rmin:.3f}, {rmax:.3f}] Mpc")

    # ── Compute ──────────────────────────────────────────────────────────────
    fields = [fs_data, s1_data, s2_data]
    xis, r_cs = [], []

    for data, name in zip(fields, labels):
        np_xy = data.shape[0] * data.shape[1]
        print(f"\nComputing LS xi for {name}  "
              f"(shape={data.shape}, n_pencils={np_xy}) ...")
        xi, r_c, DD, DR, RR = xi_ls_along_z(data, args.Lz, bin_edges)
        xis.append(xi)
        r_cs.append(r_c)
        finite = np.isfinite(xi)
        print(f"  xi ∈ [{np.nanmin(xi):.4e}, {np.nanmax(xi):.4e}]  "
              f"({finite.sum()}/{len(xi)} finite bins)")
        print(f"  RR range: [{RR.min():.0f}, {RR.max():.0f}]  "
              f"(= n_pairs × n_pencils)")

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
    ax.set_xlabel(r'$r_z$  [Mpc]', fontsize=13)

    if args.r2:
        ylabel = r'$\xi_{\rm LS}(r_z)\,r_z^2$  [Mpc$^2$]'
    else:
        ylabel = r'$\xi_{\rm LS}(r_z)$'
    if args.logy:
        ax.set_yscale('log')
        ylabel = (r'$|\xi_{\rm LS}|\,r_z^2$' if args.r2
                  else r'$|\xi_{\rm LS}(r_z)|$')
    ax.set_ylabel(ylabel, fontsize=13)

    crop_note = (f'FS cropped [{args.ref_left}]–[{args.ref_right}]'
                 if do_crop else 'no FS crop')
    ax.set_title(
        f'1-D Landy–Szalay $\\xi(r_z)$ — ppp2.c style, non-periodic\n'
        f'$L_z={args.Lz}$ Mpc,  $n_z={nz}$,  '
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
