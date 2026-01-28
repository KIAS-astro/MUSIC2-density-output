import numpy as np
import matplotlib.pyplot as plt

def read_music_field(filename):
    """Read MUSIC debug output field"""
    with open(filename, 'rb') as f:
        nx = np.fromfile(f, dtype=np.uint64, count=1)[0]
        ny = np.fromfile(f, dtype=np.uint64, count=1)[0]
        nz = np.fromfile(f, dtype=np.uint64, count=1)[0]
        
        print(f"Grid size: {nx} × {ny} × {nz}")
        
        data = np.fromfile(f, dtype=np.float64)
        data = data.reshape((nx, ny, nz))
    
    return data

# Read real-space density
delta_real = read_music_field('delta_level6_real.dat')

print(f"Statistics:")
print(f"  Mean: {np.mean(delta_real):.6e}")
print(f"  Std:  {np.std(delta_real):.6e}")
print(f"  Min:  {np.min(delta_real):.6e}")
print(f"  Max:  {np.max(delta_real):.6e}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Slice through middle
mid = delta_real.shape[2] // 2
axes[0, 0].imshow(delta_real[:, :, mid], cmap='RdBu_r', origin='lower')
axes[0, 0].set_title('Density Field δ(x) - XY slice')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# Histogram
axes[0, 1].hist(delta_real.flatten(), bins=100, density=True, alpha=0.7)
axes[0, 1].set_xlabel('δ')
axes[0, 1].set_ylabel('PDF')
axes[0, 1].set_title('Density Distribution')
axes[0, 1].set_yscale('log')

# Power spectrum (if you have the file)
try:
    pk_data = np.loadtxt('delta_level6_pk.txt')
    k = pk_data[:, 0]
    pk = pk_data[:, 1]
    
    axes[1, 0].loglog(k, pk, 'o-')
    axes[1, 0].set_xlabel('k [h/Mpc]')
    axes[1, 0].set_ylabel('P(k) [(Mpc/h)³]')
    axes[1, 0].set_title('Power Spectrum from Density Field')
    axes[1, 0].grid(True, alpha=0.3)
except:
    print("Power spectrum file not found")

# Projection
axes[1, 1].imshow(np.mean(delta_real, axis=2), cmap='RdBu_r', origin='lower')
axes[1, 1].set_title('Projected Density (mean along Z)')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')

plt.tight_layout()
plt.savefig('music_density_analysis_lv6.png', dpi=150)
plt.show()
