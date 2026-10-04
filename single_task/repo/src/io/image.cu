// Tessera - image files and image comparison.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#include "tsr/io/image.h"

namespace tsr
{
void write_ppm(const std::string &path, int width, int height, const uchar4 *pixels)
{
    std::ofstream f(path, std::ios::binary);
    TSR_CHECK(f.good(), "cannot write '" << path << "'");
    f << "P6\n" << width << " " << height << "\n255\n";
    std::vector<unsigned char> row(static_cast<std::size_t>(width) * 3);
    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const uchar4 p = pixels[static_cast<std::size_t>(y) * width + x];
            row[3 * x + 0] = p.x;
            row[3 * x + 1] = p.y;
            row[3 * x + 2] = p.z;
        }
        f.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    TSR_CHECK(f.good(), "error while writing '" << path << "'");
}

namespace
{
// Reads the next header token, skipping whitespace and '#' comments.
std::string next_token(std::istream &in)
{
    std::string tok;
    char c;
    while(in.get(c))
    {
        if(c == '#')
        {
            std::string comment;
            std::getline(in, comment);
            continue;
        }
        if(std::isspace(static_cast<unsigned char>(c)))
        {
            if(!tok.empty())
            {
                return tok;
            }
            continue;
        }
        tok.push_back(c);
    }
    return tok;
}
} // namespace

Image8 read_ppm(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    TSR_CHECK(f.good(), "cannot read '" << path << "'");
    TSR_CHECK(next_token(f) == "P6", "'" << path << "' is not a binary PPM");
    Image8 img;
    img.width            = std::atoi(next_token(f).c_str());
    img.height           = std::atoi(next_token(f).c_str());
    const int max_value  = std::atoi(next_token(f).c_str());
    TSR_CHECK(img.width > 0 && img.height > 0 && max_value == 255, "unsupported PPM header in '" << path << "'");
    std::vector<unsigned char> data(static_cast<std::size_t>(img.width) * img.height * 3);
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
    TSR_CHECK(f.gcount() == static_cast<std::streamsize>(data.size()), "truncated PPM '" << path << "'");
    img.pixels.resize(static_cast<std::size_t>(img.width) * img.height);
    for(std::size_t i = 0; i < img.pixels.size(); ++i)
    {
        img.pixels[i] = make_uchar4(data[3 * i], data[3 * i + 1], data[3 * i + 2], 255);
    }
    return img;
}

void write_pfm(const std::string &path, int width, int height, const float3 *pixels)
{
    std::ofstream f(path, std::ios::binary);
    TSR_CHECK(f.good(), "cannot write '" << path << "'");
    f << "PF\n" << width << " " << height << "\n-1.0\n";
    for(int y = height - 1; y >= 0; --y)
    {
        f.write(reinterpret_cast<const char *>(pixels + static_cast<std::size_t>(y) * width), static_cast<std::streamsize>(sizeof(float3)) * width);
    }
}

ImageDiff compare_images(const Image8 &a, const Image8 &b)
{
    TSR_CHECK(a.width == b.width && a.height == b.height, "image sizes differ: " << a.width << "x" << a.height << " vs " << b.width << "x" << b.height);
    ImageDiff d;
    double sq_sum = 0.0;
    for(std::size_t i = 0; i < a.pixels.size(); ++i)
    {
        const uchar4 p = a.pixels[i], q = b.pixels[i];
        const int dx = std::abs(int(p.x) - int(q.x)), dy = std::abs(int(p.y) - int(q.y)), dz = std::abs(int(p.z) - int(q.z));
        sq_sum         += double(dx) * dx + double(dy) * dy + double(dz) * dz;
        d.max_abs_diff  = std::max(d.max_abs_diff, std::max(dx, std::max(dy, dz)));
        d.differing_pixels += (dx | dy | dz) != 0 ? 1 : 0;
    }
    const double mse = sq_sum / (3.0 * static_cast<double>(a.pixels.size()));
    d.psnr           = mse == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(255.0 * 255.0 / mse);
    return d;
}
} // namespace tsr
