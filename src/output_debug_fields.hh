#pragma once

#include <fstream>
#include <string>
#include <complex>
#include <cmath>
#include <map>
#include <algorithm>
#include "general.hh"
#include "mesh.hh"
#include "config_file.hh"

namespace debug_output {

// Print expected refinement region from config file
inline void print_expected_refinement_region(config_file& cf, int level) {
    if (!cf.contains_key("setup", "ref_center") && !cf.contains_key("setup", "ref_offset")) {
        music::ilog << "DEBUG: No zoom-in region specified (unigrid run)" << std::endl;
        return;
    }

    double xcen[3], xext[3];
    std::string temp;

    // Read ref_center
    if (cf.contains_key("setup", "ref_center")) {
        temp = cf.get_value<std::string>("setup", "ref_center");
        std::remove_if(temp.begin(), temp.end(), isspace);
        sscanf(temp.c_str(), "%lf,%lf,%lf", &xcen[0], &xcen[1], &xcen[2]);
    }

    // Read ref_extent
    if (cf.contains_key("setup", "ref_extent")) {
        temp = cf.get_value<std::string>("setup", "ref_extent");
        std::remove_if(temp.begin(), temp.end(), isspace);
        sscanf(temp.c_str(), "%lf,%lf,%lf", &xext[0], &xext[1], &xext[2]);
    }

    // Compute expected region bounds
    double x0[3], x1[3];
    for (int i = 0; i < 3; ++i) {
        x0[i] = xcen[i] - 0.5 * xext[i];
        x1[i] = xcen[i] + 0.5 * xext[i];
    }

    music::ilog << "DEBUG: Expected refinement region from config (level " << level << "):" << std::endl;
    music::ilog << "  ref_center: (" << xcen[0] << ", " << xcen[1] << ", " << xcen[2] << ")" << std::endl;
    music::ilog << "  ref_extent: (" << xext[0] << ", " << xext[1] << ", " << xext[2] << ")" << std::endl;
    music::ilog << "  Expected box: [" << x0[0] << ":" << x1[0] << ", "
                << x0[1] << ":" << x1[1] << ", " << x0[2] << ":" << x1[2] << "]" << std::endl;
}

// Output real-space density field
template<typename GridType>
void write_density_real(const GridType& field, int level, const std::string& prefix = "delta") {
    char filename[256];
    snprintf(filename, 256, "%s_level%d_real.dat", prefix.c_str(), level);

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.good()) {
        music::wlog << "Could not open " << filename << " for writing!" << std::endl;
        return;
    }

    size_t nx = field.size(0);
    size_t ny = field.size(1);
    size_t nz = field.size(2);

    // Get offset information if available (for subgrids)
    int ox = 0, oy = 0, oz = 0;
    try {
        ox = field.offset(0);
        oy = field.offset(1);
        oz = field.offset(2);
    } catch (...) {
        // Base grid has no offset, leave as 0
    }

    // Write header with offset information
    ofs.write(reinterpret_cast<const char*>(&nx), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&ny), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&nz), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&ox), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&oy), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&oz), sizeof(int));

    // Write data
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                real_t val = field(i, j, k);
                ofs.write(reinterpret_cast<const char*>(&val), sizeof(real_t));
            }
        }
    }
    ofs.close();

    music::ilog << "DEBUG: Wrote real-space field to " << filename << std::endl;
    music::ilog << "  Grid size: " << nx << " × " << ny << " × " << nz << std::endl;
    music::ilog << "  Grid offset: (" << ox << ", " << oy << ", " << oz << ")" << std::endl;

    // Compute and display physical box coordinates
    size_t ngrid = 1ul << level;
    double dx = 1.0 / (double)ngrid;
    double x0 = ox * dx;
    double y0 = oy * dx;
    double z0 = oz * dx;
    double x1 = x0 + nx * dx;
    double y1 = y0 + ny * dx;
    double z1 = z0 + nz * dx;

    music::ilog << "  Physical region: [" << x0 << ":" << x1 << ", "
                << y0 << ":" << y1 << ", " << z0 << ":" << z1 << "]" << std::endl;
    music::ilog << "  Physical center: (" << 0.5*(x0+x1) << ", "
                << 0.5*(y0+y1) << ", " << 0.5*(z0+z1) << ")" << std::endl;
}

// Output Fourier-space density field
template<typename GridType>
void write_density_fourier(const GridType& field, int level, const std::string& prefix = "delta") {
    char filename[256];
    snprintf(filename, 256, "%s_level%d_fourier.dat", prefix.c_str(), level);
    
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.good()) {
        music::wlog << "Could not open " << filename << " for writing!" << std::endl;
        return;
    }
    
    size_t nx = field.size(0);
    size_t ny = field.size(1);
    size_t nz = field.size(2);
    size_t nzp = 2 * (nz/2 + 1);
    
    // Allocate FFT arrays
    real_t *data = new real_t[nx * ny * nzp];
    complex_t *cdata = reinterpret_cast<complex_t*>(data);
    
    // Copy data
    #pragma omp parallel for
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                size_t idx = (i * ny + j) * nzp + k;
                data[idx] = field(i, j, k);
            }
        }
    }
    
    // FFT to k-space
    fftw_plan_t plan = FFTW_API(plan_dft_r2c_3d)(nx, ny, nz, data, cdata, FFTW_ESTIMATE);
    FFTW_API(execute)(plan);
    
    // Write header
    ofs.write(reinterpret_cast<const char*>(&nx), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&ny), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&nz), sizeof(size_t));
    
    // Write complex data
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz/2 + 1; ++k) {
                size_t idx = (i * ny + j) * (nzp/2) + k;
                real_t re = RE(cdata[idx]);
                real_t im = IM(cdata[idx]);
                ofs.write(reinterpret_cast<const char*>(&re), sizeof(real_t));
                ofs.write(reinterpret_cast<const char*>(&im), sizeof(real_t));
            }
        }
    }
    
    FFTW_API(destroy_plan)(plan);
    delete[] data;
    ofs.close();
    
    music::ilog << "DEBUG: Wrote Fourier-space field to " << filename << std::endl;
}

// Compute and output power spectrum
template<typename GridType>
void write_power_spectrum(const GridType& field, int level, real_t boxsize, 
                         const std::string& prefix = "delta") {
    char filename[256];
    snprintf(filename, 256, "%s_level%d_pk.txt", prefix.c_str(), level);
    
    std::ofstream ofs(filename);
    if (!ofs.good()) {
        music::wlog << "Could not open " << filename << " for writing!" << std::endl;
        return;
    }
    
    size_t nx = field.size(0);
    size_t ny = field.size(1);
    size_t nz = field.size(2);
    size_t nzp = 2 * (nz/2 + 1);
    
    // Allocate and FFT
    real_t *data = new real_t[nx * ny * nzp];
    complex_t *cdata = reinterpret_cast<complex_t*>(data);
    
    #pragma omp parallel for
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                size_t idx = (i * ny + j) * nzp + k;
                data[idx] = field(i, j, k);
            }
        }
    }
    
    fftw_plan_t plan = FFTW_API(plan_dft_r2c_3d)(nx, ny, nz, data, cdata, FFTW_ESTIMATE);
    FFTW_API(execute)(plan);
    
    // Bin power spectrum
    std::map<int, std::pair<double, int>> pk_bins;
    
    double kfund = 2.0 * M_PI / boxsize;
    double norm = 1.0 / ((double)nx * (double)ny * (double)nz);
    
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz/2 + 1; ++k) {
                int ii = (i > nx/2) ? (int)i - (int)nx : (int)i;
                int jj = (j > ny/2) ? (int)j - (int)ny : (int)j;
                int kk = (int)k;
                
                double kmag = kfund * std::sqrt((double)(ii*ii + jj*jj + kk*kk));
                
                if (kmag > 0) {
                    size_t idx = (i * ny + j) * (nzp/2) + k;
                    double power = (RE(cdata[idx])*RE(cdata[idx]) + 
                                   IM(cdata[idx])*IM(cdata[idx])) * norm * norm;
                    
                    int kbin = (int)(kmag / kfund + 0.5);
                    pk_bins[kbin].first += power;
                    pk_bins[kbin].second += 1;
                }
            }
        }
    }
    
    // Write
    ofs << "# k [h/Mpc]    P(k) [(Mpc/h)^3]    N_modes\n";
    ofs << "# Box size: " << boxsize << " Mpc/h\n";
    ofs << "# Grid: " << nx << " × " << ny << " × " << nz << "\n";
    ofs << "# Fundamental k: " << kfund << " h/Mpc\n";
    
    for (const auto& bin : pk_bins) {
        double k = bin.first * kfund;
        double pk = bin.second.first / bin.second.second;
        int nmodes = bin.second.second;
        ofs << k << "  " << pk << "  " << nmodes << "\n";
    }
    
    FFTW_API(destroy_plan)(plan);
    delete[] data;
    ofs.close();
    
    music::ilog << "DEBUG: Wrote power spectrum to " << filename << std::endl;
}

// Output statistics
template<typename GridType>
void print_field_statistics(const GridType& field, const std::string& name) {
    size_t nx = field.size(0);
    size_t ny = field.size(1);
    size_t nz = field.size(2);
    
    double mean = 0.0, var = 0.0, minval = 1e30, maxval = -1e30;
    
    #pragma omp parallel for reduction(+:mean,var) reduction(min:minval) reduction(max:maxval)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                double val = field(i, j, k);
                mean += val;
                var += val * val;
                minval = std::min(minval, val);
                maxval = std::max(maxval, val);
            }
        }
    }
    
    size_t ntotal = nx * ny * nz;
    mean /= ntotal;
    var = var / ntotal - mean * mean;
    double std = std::sqrt(var);
    
    music::ilog << "DEBUG: Statistics for " << name << ":" << std::endl;
    music::ilog << "  Mean:   " << mean << std::endl;
    music::ilog << "  Std:    " << std << std::endl;
    music::ilog << "  Min:    " << minval << std::endl;
    music::ilog << "  Max:    " << maxval << std::endl;
}

} // namespace debug_output
