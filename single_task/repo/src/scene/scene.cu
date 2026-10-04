// Tessera - scene description: validation, the .tsr text format and
// synthetic scenes.
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tsr/math/vec.h"
#include "tsr/scene/scene.h"
#include "tsr/util/random.h"

namespace tsr
{
void GaussianCloud::validate() const
{
    const std::size_t n = means.size();
    TSR_CHECK(quats.size() == n && scales.size() == n && opacities.size() == n,
              "Gaussian arrays disagree: means " << n << ", quats " << quats.size() << ", scales " << scales.size()
                                                 << ", opacities " << opacities.size());
    TSR_CHECK(sh_degree <= 3, "SH degree " << sh_degree << " is not supported (max 3)");
    TSR_CHECK(sh.size() == n * coeffs_per_gaussian() * 3,
              "expected " << n * coeffs_per_gaussian() * 3 << " SH coefficients for degree " << sh_degree << ", got " << sh.size());
    for(std::size_t i = 0; i < n; ++i)
    {
        TSR_CHECK(opacities[i] >= 0.f && opacities[i] <= 1.f, "Gaussian " << i << " has opacity " << opacities[i] << " outside [0, 1]");
        TSR_CHECK(scales[i].x > 0.f && scales[i].y > 0.f && scales[i].z > 0.f, "Gaussian " << i << " has a non-positive scale");
    }
}

std::size_t Scene::face_count() const
{
    std::size_t total = 0;
    for(const auto &m : meshes)
    {
        total += m.faces.size();
    }
    return total;
}

std::size_t Scene::vertex_count() const
{
    std::size_t total = 0;
    for(const auto &m : meshes)
    {
        total += m.vertices.size();
    }
    return total;
}

namespace
{
class LineReader
{
public:
    LineReader(const std::string &source, std::size_t line, const std::string &text)
        : source_(source), line_(line), stream_(text)
    {
    }

    template<typename T>
    T next(const char *what)
    {
        T value{};
        if(!(stream_ >> value))
        {
            fail(std::string("expected ") + what);
        }
        return value;
    }

    float3 next_float3(const char *what)
    {
        const float x = next<float>(what);
        const float y = next<float>(what);
        const float z = next<float>(what);
        return make_float3(x, y, z);
    }

    bool try_word(std::string &word) { return static_cast<bool>(stream_ >> word); }

    void expect_end()
    {
        std::string rest;
        if(stream_ >> rest)
        {
            fail("unexpected trailing token '" + rest + "'");
        }
    }

    [[noreturn]] void fail(const std::string &msg) const
    {
        std::ostringstream os;
        os << source_ << ":" << line_ << ": " << msg;
        throw Error(os.str());
    }

private:
    const std::string &source_;
    std::size_t line_;
    std::istringstream stream_;
};

Camera parse_camera(LineReader &r)
{
    const std::string kind = r.next<std::string>("camera kind (lookat|ortho)");
    if(kind != "lookat" && kind != "ortho")
    {
        r.fail("unknown camera kind '" + kind + "'");
    }
    const float3 eye    = r.next_float3("eye position");
    const float3 target = r.next_float3("target position");
    const float3 up     = r.next_float3("up vector");
    float fov = 45.f, scale = 0.f, near_plane = 0.01f, far_plane = 1e10f;
    std::uint32_t width = 0, height = 0;
    std::string key;
    while(r.try_word(key))
    {
        if(key == "fov")
        {
            fov = r.next<float>("field of view");
        }
        else if(key == "scale")
        {
            scale = r.next<float>("orthographic scale");
        }
        else if(key == "size")
        {
            width  = r.next<std::uint32_t>("width");
            height = r.next<std::uint32_t>("height");
        }
        else if(key == "near")
        {
            near_plane = r.next<float>("near plane");
        }
        else if(key == "far")
        {
            far_plane = r.next<float>("far plane");
        }
        else
        {
            r.fail("unknown camera option '" + key + "'");
        }
    }
    if(width == 0 || height == 0)
    {
        r.fail("camera needs 'size <width> <height>'");
    }
    Camera cam;
    try
    {
        cam = Camera::look_at(eye, target, up, fov, width, height);
    }
    catch(const Error &e)
    {
        r.fail(e.what());
    }
    cam.near_plane = near_plane;
    cam.far_plane  = far_plane;
    if(kind == "ortho")
    {
        if(scale <= 0.f)
        {
            r.fail("orthographic camera needs 'scale <pixels per unit>'");
        }
        cam.model = CameraModel::Orthographic;
        cam.fx = cam.fy = scale;
    }
    return cam;
}
} // namespace

Scene parse_scene(const std::string &text, const std::string &source_name)
{
    Scene scene;
    std::istringstream in(text);
    std::string line;
    std::size_t line_no      = 0;
    bool header_seen         = false;
    TriangleMesh *open_mesh  = nullptr;
    bool sh_degree_locked    = false;

    while(std::getline(in, line))
    {
        ++line_no;
        const auto hash = line.find('#');
        if(hash != std::string::npos)
        {
            line.erase(hash);
        }
        LineReader r(source_name, line_no, line);
        std::string cmd;
        if(!r.try_word(cmd))
        {
            continue;
        }
        if(!header_seen)
        {
            if(cmd != "tessera-scene")
            {
                r.fail("missing 'tessera-scene <version>' header");
            }
            const int version = r.next<int>("format version");
            if(version != 1)
            {
                r.fail("unsupported scene format version " + std::to_string(version));
            }
            header_seen = true;
            continue;
        }

        if(open_mesh != nullptr)
        {
            if(cmd == "v")
            {
                open_mesh->vertices.push_back(r.next_float3("vertex position"));
                r.expect_end();
            }
            else if(cmd == "f")
            {
                // 1-based indices, as in Wavefront OBJ
                const int a = r.next<int>("face index");
                const int b = r.next<int>("face index");
                const int c = r.next<int>("face index");
                r.expect_end();
                const int nv = static_cast<int>(open_mesh->vertices.size());
                if(a < 1 || b < 1 || c < 1 || a > nv || b > nv || c > nv)
                {
                    r.fail("face index out of range (vertices are numbered from 1)");
                }
                open_mesh->faces.push_back(make_int3(a - 1, b - 1, c - 1));
            }
            else if(cmd == "end")
            {
                r.expect_end();
                open_mesh = nullptr;
            }
            else
            {
                r.fail("unexpected '" + cmd + "' inside mesh block (missing 'end'?)");
            }
            continue;
        }

        if(cmd == "background")
        {
            scene.background = r.next_float3("background colour");
            r.expect_end();
        }
        else if(cmd == "camera")
        {
            scene.cameras.push_back(parse_camera(r));
        }
        else if(cmd == "sh_degree")
        {
            if(sh_degree_locked)
            {
                r.fail("sh_degree must come before the first gaussian");
            }
            scene.gaussians.sh_degree = r.next<std::uint32_t>("SH degree");
            r.expect_end();
            if(scene.gaussians.sh_degree > 3)
            {
                r.fail("SH degree must be <= 3");
            }
        }
        else if(cmd == "gaussian")
        {
            sh_degree_locked = true;
            GaussianCloud &g = scene.gaussians;
            g.means.push_back(r.next_float3("mean"));
            const float qw = r.next<float>("quaternion");
            const float qx = r.next<float>("quaternion");
            const float qy = r.next<float>("quaternion");
            const float qz = r.next<float>("quaternion");
            g.quats.push_back(make_float4(qw, qx, qy, qz));
            g.scales.push_back(r.next_float3("scale"));
            g.opacities.push_back(r.next<float>("opacity"));
            const std::uint32_t coeffs = g.coeffs_per_gaussian() * 3;
            for(std::uint32_t k = 0; k < coeffs; ++k)
            {
                g.sh.push_back(r.next<float>("SH coefficient"));
            }
            r.expect_end();
        }
        else if(cmd == "mesh")
        {
            scene.meshes.emplace_back();
            open_mesh       = &scene.meshes.back();
            open_mesh->name = r.next<std::string>("mesh name");
            r.expect_end();
        }
        else if(cmd == "material")
        {
            NeuralMaterial &m = scene.material;
            std::string key;
            while(r.try_word(key))
            {
                if(key == "grid")
                {
                    m.grid.n_levels           = r.next<std::uint32_t>("levels");
                    m.grid.features_per_level = r.next<std::uint32_t>("features per level");
                    m.grid.log2_hashmap_size  = r.next<std::uint32_t>("log2 hashmap size");
                    m.grid.base_resolution    = r.next<std::uint32_t>("base resolution");
                    m.grid.per_level_scale    = r.next<float>("per-level scale");
                }
                else if(key == "mlp")
                {
                    m.mlp.hidden_width  = r.next<std::uint32_t>("hidden width");
                    m.mlp.hidden_layers = r.next<std::uint32_t>("hidden layers");
                }
                else if(key == "seed")
                {
                    m.seed = r.next<std::uint32_t>("seed");
                }
                else if(key == "bounds")
                {
                    m.bounds_min = r.next_float3("bounds min");
                    m.bounds_max = r.next_float3("bounds max");
                }
                else if(key == "tint")
                {
                    m.tint = r.next_float3("tint");
                }
                else
                {
                    r.fail("unknown material option '" + key + "'");
                }
            }
        }
        else
        {
            r.fail("unknown command '" + cmd + "'");
        }
    }
    if(!header_seen)
    {
        throw Error(source_name + ": empty scene file");
    }
    if(open_mesh != nullptr)
    {
        throw Error(source_name + ": mesh '" + open_mesh->name + "' is missing its 'end'");
    }
    scene.gaussians.validate();
    return scene;
}

Scene load_scene(const std::string &path)
{
    std::ifstream f(path);
    TSR_CHECK(f.good(), "cannot open scene file '" << path << "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_scene(ss.str(), path);
}

Scene make_synthetic_scene(const SyntheticSceneParams &params)
{
    Pcg32 rng(params.seed, 0x2545f4914f6cdd1dULL);
    Scene scene;
    scene.background = make_float3(0.05f, 0.06f, 0.09f);

    GaussianCloud &g = scene.gaussians;
    g.sh_degree      = params.sh_degree;
    const std::uint32_t K = g.coeffs_per_gaussian();
    for(std::uint32_t i = 0; i < params.n_gaussians; ++i)
    {
        g.means.push_back(make_float3(rng.uniform(-1.2f, 1.2f), rng.uniform(-0.8f, 1.0f), rng.uniform(-1.2f, 1.2f)));
        float4 q = make_float4(rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f), rng.uniform(-1.f, 1.f));
        if(std::fabs(q.x) + std::fabs(q.y) + std::fabs(q.z) + std::fabs(q.w) < 1e-3f)
        {
            q = make_float4(1.f, 0.f, 0.f, 0.f);
        }
        g.quats.push_back(q);
        const float base = std::exp(rng.uniform(std::log(0.02f), std::log(0.12f)));
        g.scales.push_back(make_float3(base * rng.uniform(0.5f, 2.f), base * rng.uniform(0.5f, 2.f), base * rng.uniform(0.5f, 2.f)));
        g.opacities.push_back(rng.uniform(0.25f, 0.95f));
        for(std::uint32_t k = 0; k < K; ++k)
        {
            const float amp = k == 0 ? 1.0f : 0.25f;
            for(int c = 0; c < 3; ++c)
            {
                g.sh.push_back(rng.uniform(-amp, amp));
            }
        }
    }

    if(params.with_mesh)
    {
        // A floor quad and an axis-aligned box that occludes part of the cloud.
        TriangleMesh floor;
        floor.name     = "floor";
        floor.vertices = {make_float3(-2.f, -1.f, -2.f), make_float3(2.f, -1.f, -2.f), make_float3(2.f, -1.f, 2.f), make_float3(-2.f, -1.f, 2.f)};
        floor.faces    = {make_int3(0, 2, 1), make_int3(0, 3, 2)};
        scene.meshes.push_back(floor);

        TriangleMesh box;
        box.name        = "box";
        const float h   = 0.35f;
        const float3 c  = make_float3(0.2f, -0.4f, 0.1f);
        for(int i = 0; i < 8; ++i)
        {
            box.vertices.push_back(make_float3(c.x + ((i & 1) ? h : -h), c.y + ((i & 2) ? h : -h), c.z + ((i & 4) ? h : -h)));
        }
        box.faces = {
            make_int3(0, 1, 3), make_int3(0, 3, 2), // -z
            make_int3(4, 6, 7), make_int3(4, 7, 5), // +z
            make_int3(0, 4, 5), make_int3(0, 5, 1), // -y
            make_int3(2, 3, 7), make_int3(2, 7, 6), // +y
            make_int3(0, 2, 6), make_int3(0, 6, 4), // -x
            make_int3(1, 5, 7), make_int3(1, 7, 3), // +x
        };
        scene.meshes.push_back(box);
    }

    scene.material.seed       = params.seed * 7919u + 17u;
    scene.material.bounds_min = make_float3(-2.f, -1.5f, -2.f);
    scene.material.bounds_max = make_float3(2.f, 1.5f, 2.f);
    scene.material.tint       = make_float3(0.9f, 0.85f, 0.8f);

    scene.cameras.push_back(Camera::look_at(make_float3(0.4f, 0.9f, -3.6f), make_float3(0.f, -0.2f, 0.f), make_float3(0.f, 1.f, 0.f), 50.f, params.width, params.height));
    scene.cameras.push_back(Camera::look_at(make_float3(-3.0f, 0.5f, 1.5f), make_float3(0.f, -0.3f, 0.f), make_float3(0.f, 1.f, 0.f), 45.f, params.width, params.height));
    return scene;
}
} // namespace tsr
