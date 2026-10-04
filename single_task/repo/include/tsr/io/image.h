// Tessera - image files and image comparison.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tsr/device/backend.h"

namespace tsr
{
struct Image8
{
    int width  = 0;
    int height = 0;
    std::vector<uchar4> pixels; // row-major, top row first
};

// Binary PPM (P6), alpha dropped on write and set to 255 on read.
void write_ppm(const std::string &path, int width, int height, const uchar4 *pixels);
Image8 read_ppm(const std::string &path);

// Portable float map (PF), little endian. PFM stores the bottom row first;
// the writer flips so that `pixels` stays top row first.
void write_pfm(const std::string &path, int width, int height, const float3 *pixels);

struct ImageDiff
{
    double psnr          = 0.0; // dB over RGB, infinity for identical images
    int max_abs_diff     = 0;   // largest channel difference
    std::size_t differing_pixels = 0;
};

ImageDiff compare_images(const Image8 &a, const Image8 &b);

} // namespace tsr
