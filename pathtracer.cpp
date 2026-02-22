#include "pathtracer.h"

#include <iostream>

#include <scene/shape/Sphere.h>

#include <random>

#include <Eigen/Dense>

#include <fstream>
#include <string>

#include <util/Common.h>

using namespace Eigen;

PathTracer::PathTracer(int width, int height)
    : m_width(width), m_height(height)
{
}

float luminance(const Vector3f& v)
{
    return 0.2126f * v.x() +
           0.7152f * v.y() +
           0.0722f * v.z();
}

static thread_local std::mt19937 generator{std::random_device{}()};
static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

float getRandom() {
    return distribution(generator);
}

// this is low discrepancy sampling
float vanDerCorput(int n, int base)
{
    float invBase = 1.f / base;
    float denom = invBase;
    float result = 0;
    while (n > 0) {
        result += (n % base) * denom;
        n /= base;
        denom *= invBase;
    }
    return result;
}

PixelGeometry getPrimaryGeometry(const Ray& r, const Scene& scene) {
    IntersectionInfo p;
    PixelGeometry geo;
    geo.color = Vector3f(0,0,0);
    geo.normal = Vector3f(0,0,0);
    geo.depth = 1e10f;
    geo.albedo = Vector3f(0,0,0);

    if(scene.getIntersection(r, &p)) {
        const Triangle *t = static_cast<const Triangle *>(p.data);
        const tinyobj::material_t& mat = t->getMaterial();

        geo.normal = t->getNormal(p);
        geo.depth = p.t;
        geo.albedo = Vector3f(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
    }
    return geo;
}

float localVariance(int x, int y, const std::vector<RenderResult>& r_res, int width, int height) {
    float sumLum = 0.0f;
    float sumLumSq = 0.0f;
    int count = 0;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int qx = std::clamp(x + i, 0, width - 1);
            int qy = std::clamp(y + j, 0, height - 1);
            float l = luminance(r_res[qx + qy * width].radiance);
            sumLum += l;
            sumLumSq += l * l;
            count++;
        }
    }

    float mean = sumLum / count;
    float var = (sumLumSq / count) - (mean * mean);
    return std::max(0.0f, var);
}

// https://jo.dreggn.org/home/2010_atrous.pdf
float calculateWeight(const RenderResult& p, const RenderResult& q,
                                  const Vector2f& p_pos, const Vector2f& q_pos,
                                  float p_var, const Vector2f& p_grad_z, int step) {
    // Normal
    float c_phi = 1.0f;
    float n_phi = 0.1f;
    float p_phi = 0.5f;
    Vector3f n_diff = p.normal - q.normal;
    float n_dist2 = n_diff.dot(n_diff);

    float n_w = std::min(std::exp(-(n_dist2) / n_phi), 1.0f);

    // Luminance
    float p_lum = luminance(p.radiance);
    float q_lum = luminance(q.radiance);
    float l_diff = p_lum - q_lum;
    float l_dist2 = l_diff * l_diff;
    float c_w = std::min(std::exp(-(l_dist2) / c_phi), 1.0f);

    // Depth
    float z_diff = p.depth - q.depth;
    float z_dist2 = z_diff * z_diff;
    float p_w = std::min(std::exp(-(z_dist2) / p_phi), 1.0f);

    return c_w * n_w * p_w;
}

void aTrousWavelet(std::vector<RenderResult>& r_res, int width, int height) {
    const float h[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

    std::vector<float> varianceMap(width * height);
    std::vector<Vector2f> depthGradMap(width * height);

// #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = x + y * width;

            // depth
            int x_next = std::min(x + 1, width - 1);
            int y_next = std::min(y + 1, height - 1);
            float dz_dx = r_res[x_next + y * width].depth - r_res[idx].depth;
            float dz_dy = r_res[x + y_next * width].depth - r_res[idx].depth;
            depthGradMap[idx] = Vector2f(dz_dx, dz_dy);

            // variance
            varianceMap[idx] = localVariance(x, y, r_res, width, height);
        }
    }

    std::vector<Vector3f> currentRadiance(width * height);
    for(int i = 0; i < width * height; ++i) currentRadiance[i] = r_res[i].radiance;
    std::vector<Vector3f> nextRadiance(width * height);

    for (int pass = 0; pass < 3; ++pass) {
        int step = 1 << pass;

// #pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vector3f sumColor(0, 0, 0);
                float sumWeight = 0.0f;
                const RenderResult& p_geo = r_res[x + y * width];
                float p_var = varianceMap[x + y * width];
                Vector2f p_grad = depthGradMap[x + y * width];

                for (int i = -2; i <= 2; ++i) {
                    for (int j = -2; j <= 2; ++j) {
                        int qx = std::clamp(x + i * step, 0, width - 1);
                        int qy = std::clamp(y + j * step, 0, height - 1);

                        const RenderResult& q_geo = r_res[qx + qy * width];
                        float kernelW = h[i + 2] * h[j + 2];

                        float edgeW = calculateWeight(
                            p_geo, q_geo,
                            Vector2f(x, y), Vector2f(qx, qy),
                            p_var, p_grad, step
                            );

                        float totalW = kernelW * edgeW;
                        sumColor += currentRadiance[qx + qy * width] * totalW;
                        sumWeight += totalW;
                    }
                }
                nextRadiance[x + y * width] = sumColor / (sumWeight + 1e-6f);
            }
        }
        currentRadiance = nextRadiance;
    }

    for(int i = 0; i < width * height; ++i) r_res[i].radiance = currentRadiance[i];
}

// add stratified + low discrepency sampling
void PathTracer::traceScene(QRgb *imageData, const Scene& scene)
{
    std::string scene_name = "test10";
    bool is_ground_truth = (settings.samplesPerPixel > 128);
    std::vector<Vector3f> intensityValues(m_width * m_height, Vector3f(0,0,0));
    Matrix4f invViewMat = (scene.getCamera().getScaleMatrix() * scene.getCamera().getViewMatrix()).inverse();

    int samplesPerPixel = settings.samplesPerPixel;
    std::vector<RenderResult> r_res(m_width * m_height);
    for(int y = 0; y < m_height; ++y) {
#pragma omp parallel for
        for(int x = 0; x < m_width; ++x) {
            Vector3f accumulatedRadiance(0,0,0);
            RenderResult first_vals;
            Vector3f avgNormal(0,0,0);
            float avgDepth = 0.0f;
            for(int s = 0; s < samplesPerPixel; ++s) {
                float jitterX = x + (getRandom() - 0.5f);
                float jitterY = y + (getRandom() - 0.5f);
                RenderResult res = tracePixel(jitterX, jitterY, scene, invViewMat);
                // if (s == 0) {
                //     r_res[x + (y * m_width)].albedo = res.albedo;
                //     r_res[x + (y * m_width)].depth = res.depth;
                //     r_res[x + (y * m_width)].normal = res.normal;
                // }

                avgNormal += res.normal;
                avgDepth += res.depth;

                accumulatedRadiance += res.radiance;
            }
            intensityValues[x + (y * m_width)] = accumulatedRadiance / samplesPerPixel;
            r_res[x + (y * m_width)].normal = avgNormal.normalized();
            r_res[x + (y * m_width)].depth = avgDepth / samplesPerPixel;
            r_res[x + (y * m_width)].radiance = intensityValues[x + (y * m_width)];

            // int n = std::sqrt(samplesPerPixel);
            // float invN = 1.0f / n;

            // for(int p = 0; p < n; ++p) {
            //     for(int q = 0; q < n; ++q) {
            //         // Stratified jitter: (fixed grid offset) + (random sub-grid jitter)
            //         float jitterX = x + (p + getRandom()) * invN - 0.5f;
            //         float jitterY = y + (q + getRandom()) * invN - 0.5f;

            //         accumulatedRadiance += tracePixel(jitterX, jitterY, scene, invViewMat);
            //     }
            // }
            // intensityValues[x + (y * m_width)] = accumulatedRadiance / (float)(n * n);
        }
    }
    // aTrousWavelet(r_res, m_width, m_height);
    // for (int i = 0; i < r_res.size(); i++) {
    //     intensityValues[i] = r_res[i].radiance;
    // }
    // data export
    auto save_binary = [&](std::string suffix, auto extractor) {
        std::string folder = (!is_ground_truth) ? "train_data/" : "test_data/";
        std::string filename = folder + scene_name + suffix + ".bin";
        std::ofstream out(filename, std::ios::binary);
        for(const auto& res : r_res) {
            auto val = extractor(res);
            out.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
        out.close();
    };

    if (is_ground_truth) {
        save_binary("_gt", [](const RenderResult& r) { return r.radiance; });
    } else {
        save_binary("_noisy", [](const RenderResult& r) { return r.radiance; });
        save_binary("_normal", [](const RenderResult& r) { return r.normal; });
        save_binary("_depth", [](const RenderResult& r) { return r.depth; });
    }

    // if (!is_ground_truth) {
    //     save_binary("_atrous", [](const RenderResult& r) { return r.radiance; });
    // }
    if (!is_ground_truth) {
        aTrousWavelet(r_res, m_width, m_height);
    }
    toneMap(imageData, intensityValues);
}

Vector3f reflect(const Vector3f& v, const Vector3f& axis) {
    return v - 2.0f * v.dot(axis) * axis;
}

const Vector3f PathTracer::calculateDirectLighting(const IntersectionInfo& p, const Vector3f& N,
                                             const Vector3f& p_d, const Vector3f& p_s,
                                            const Scene& scene, const Ray& r, tinyobj::material_t mat,
                                            const Triangle *t) {
    Vector3f L_direct(0.0f, 0.0f, 0.0f);
    const std::vector<Triangle*>& emissives = scene.getEmissives();

    if (emissives.empty()) return L_direct;

    int numEmissives = emissives.size();
    int Nsamples = settings.numDirectLightingSamples;

    for (int s = 0; s < Nsamples; ++s) {
        Triangle* tri = emissives[rand() % numEmissives];
        if (tri == t) {
            continue;
        }

        auto verts = tri->getVertices();
        Vector3f v1 = verts[0];
        Vector3f v2 = verts[1];
        Vector3f v3 = verts[2];

        float u = getRandom();
        float v = getRandom();
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        float w = 1.0f - u - v;

        Vector3f samplePos = u * v1 + v * v2 + w * v3;

        Vector3f lightNormal = tri->getNormal(samplePos);

        Vector3f di = samplePos - p.hit;
        float distSq = di.squaredNorm();
        float dist = std::sqrt(distSq);
        Vector3f wi = di / dist;

        float cosTheta = std::max(0.0f, N.dot(wi));
        float cosThetaLight = std::max(0.0f, lightNormal.dot(-wi));

        if (cosTheta <= 0.0f || cosThetaLight <= 0.0f) {
            continue;
        }

        Ray shadowRay(p.hit + N * 0.001f, wi);
        IntersectionInfo shadowHit;
        if (scene.getIntersection(shadowRay, &shadowHit) && shadowHit.t < (dist - 0.001f*2)) {
            continue;
        }

        Vector3f wo = -r.d.normalized();

        Vector3f R = reflect(-wi, N);

        float cos_alpha = std::max(0.0f, wo.dot(R));

        tinyobj::material_t lightMat = tri->getMaterial();

        Vector3f brdf = (p_d / M_PI) +
                        (p_s * ((mat.shininess + 2.0f) / (2.0f * M_PI)) * std::pow(cos_alpha, mat.shininess));

        float area = 0.5f * (v2 - v1).cross(v3 - v1).norm();

        float pdf = (1.0f / numEmissives) * (1.0f / area);

        Vector3f Le(lightMat.emission[0], lightMat.emission[1], lightMat.emission[2]);

        Vector3f contribution = Le.cwiseProduct(brdf) * cosTheta * cosThetaLight * numEmissives * area / distSq;

        // std::cout << "contribution: " << contribution << std::endl;

        L_direct += Le.cwiseProduct(brdf) * cosTheta * cosThetaLight * numEmissives * area / distSq;
    }

    // return L_direct / (float)Nsamples;
    return L_direct / (float)Nsamples;
}

Vector3f applyBeersLaw(Vector3f current_intensity, float distance, Vector3f absorption_coeff) {
    Vector3f attenuation;
    attenuation.x() = std::exp(-absorption_coeff.x() * distance);
    attenuation.y() = std::exp(-absorption_coeff.y() * distance);
    attenuation.z() = std::exp(-absorption_coeff.z() * distance);

    return current_intensity.cwiseProduct(attenuation);
}

Vector3f PathTracer::traceRay(const Ray& r, const Scene& scene, bool count_emitted)
{
    IntersectionInfo p;
    if(scene.getIntersection(r, &p)) {
        const Triangle *t = static_cast<const Triangle *>(p.data);

        const tinyobj::material_t& mat = t->getMaterial();

        Vector3f N = t->getNormal(p);

        Vector3f emitted(mat.emission[0], mat.emission[1], mat.emission[2]);

        Vector3f p_d(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);

        Vector3f p_s(mat.specular[0], mat.specular[1], mat.specular[2]);

        for (int i = 0; i < 3; ++i) {
            float sum = p_d[i] + p_s[i];

            if (sum > 1.0f) { // for glossy tests
                p_d[i] = 0.0f;
            }
        }

        Vector3f L = count_emitted ? emitted : Vector3f(0.0f,0.0f,0.0f);

        if (mat.illum != 7 && mat.illum != 5) {
            L += calculateDirectLighting(p, N, p_d, p_s, scene, r, mat, t);
        }

        float pdf_rr = settings.pathContinuationProb;
        Vector3f wi;
        // pdf_rr = 0.0f;
        if (getRandom() < pdf_rr) {
            if (mat.illum == 5) {
                // Reflect
                wi = reflect(r.d, N).normalized();
                Ray nextRay(p.hit + N * 0.0001f, wi);
                Vector3f incomingRadiance = traceRay(nextRay, scene, true);
                L += incomingRadiance / pdf_rr;
            } else if (mat.illum == 7) {
                // Refract
                Vector3f w_i = -r.d.normalized();
                float ior_air = 1.0f;
                float ior_mat = mat.ior;

                bool entering = r.d.dot(N) < 0;
                float n1 = entering ? ior_air : ior_mat;
                float n2 = entering ? ior_mat : ior_air;

                Vector3f orienting_N = entering ? N : -N;
                float eta = n1 / n2;

                float cos_theta_i = std::min(w_i.dot(orienting_N), 1.0f);
                float rad = 1.0f - std::pow(eta, 2) * (1.0f - std::pow(cos_theta_i, 2));

                // Schlick
                float r0 = std::pow((n1 - n2) / (n1 + n2), 2);
                float p_refl = r0 + (1.0f - r0) * std::pow(1.0f - cos_theta_i, 5);

                Vector3f nextDir;
                float offset_multiplier;

                if (rad < 0.0f || getRandom() < p_refl) {
                    // REFLECT
                    nextDir = reflect(r.d, orienting_N).normalized();
                    offset_multiplier = 1.0f;
                } else {
                    // REFRACT
                    float cos_theta_t = std::sqrt(rad);
                    nextDir = eta * r.d + (eta * cos_theta_i - cos_theta_t) * orienting_N;
                    offset_multiplier = -1.0f;
                }

                // Beer's law
                // Vector3f attenuation(1.0f, 1.0f, 1.0f);
                // if (!entering) {
                //     float dist = (p.hit - r.o).norm();

                    // red
                    // Vector3f absorption(0.1f, 4.0f, 4.0f);

                    // green
                    // Vector3f absorption(1.5f, 0.2f, 1.2f);

                    // blue
                    // Vector3f absorption(0.2f, 0.05f, 0.01f);

                //     attenuation.x() = std::exp(-absorption[0] * dist);
                //     attenuation.y() = std::exp(-absorption[1] * dist);
                //     attenuation.z() = std::exp(-absorption[2] * dist);
                // }

                Ray nextRay(p.hit + (orienting_N * 0.0001f * offset_multiplier), nextDir);

                L += traceRay(nextRay, scene, true) / pdf_rr;
            } else {
                // Diffuse and Glossy
                if (settings.directLightingOnly == false) {
                    float u1 = getRandom();
                    float u2 = getRandom();

                    float theta = 2.0f * M_PI * u1;
                    // float phi = acos(u2);
                    float cos_phi = u2;
                    float sin_phi = sqrt(1.0f - cos_phi * cos_phi);

                    Vector3f local_wi(sin_phi * cos(theta), sin_phi * sin(theta), cos_phi);

                    Vector3f up = (std::abs(N.x()) > 0.9f) ? Vector3f(0, 1, 0) : Vector3f(1, 0, 0);
                    Vector3f tangent = N.cross(up).normalized();
                    Vector3f bitangent = N.cross(tangent);
                    wi = (local_wi.x() * tangent + local_wi.y() * bitangent + local_wi.z() * N).normalized();

                    float pdf = 1.0f / (2.0f * M_PI);
                    float cos_term = std::max(0.0f, wi.dot(N));
                    // float cos_term = wi.dot(N);

                    Vector3f wo = -r.d.normalized();
                    Vector3f R_view = reflect(-wo, N);

                    // float cos_alpha = R_view.dot(wi);
                    float cos_alpha = std::max(0.0f, R_view.dot(wi));
                    // float cos_alpha = -wo.dot(R_view);
                    Vector3f brdf = (p_d / M_PI);

                    if (cos_alpha > 0) {
                        brdf += (p_s * ((mat.shininess + 2.0f)/(2.0f * M_PI)) * std::pow(cos_alpha, mat.shininess));
                    }

                    Ray nextRay(p.hit + N * 0.001f, wi);

                    Vector3f incomingRadiance = traceRay(nextRay, scene, false);
                    L += incomingRadiance.cwiseProduct(brdf) * cos_term / (pdf * pdf_rr);
                }
            }
        }

        return L;
    } else {
        return Vector3f(0.0f, 0.0f, 0.0f);
    }
}

RenderResult PathTracer::tracePixel(float x, float y, const Scene& scene, const Matrix4f &invViewMatrix)
{
    Vector3f p(0, 0, 0);
    Vector3f d((2.f * x / m_width) - 1, 1 - (2.f * y / m_height), -1);
    d.normalize();

    Ray r(p, d);
    r = r.transform(invViewMatrix);

    RenderResult result;
    IntersectionInfo i;
    if (scene.getIntersection(r, &i)) {
        const Triangle *t = static_cast<const Triangle *>(i.data);
        const tinyobj::material_t& mat = t->getMaterial();

        result.normal = t->getNormal(i);
        result.depth = i.t;
        result.albedo = Vector3f(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
    } else {
        result.normal = Vector3f(0, 0, 0);
        result.depth = 1e10f;
        result.albedo = Vector3f(0, 0, 0);
    }

    result.radiance = traceRay(r, scene, true);

    return result;
}

Vector3f change_luminance(Vector3f c_in, float l_out)
{
    float l_in = luminance(c_in);
    // if (l_in < 0.001 && l_in != 0.0f) {
    //     std::cout << l_in << std::endl;
    // }
    float safe_l = std::max(l_in, 0.001f);
    return c_in * (l_out / safe_l);
}

Vector3f reinhardPerChannel(const Vector3f& v, float max_white)
{
    float max_white_sq = max_white * max_white;
    Vector3f numerator = v.cwiseProduct(Vector3f::Ones() + (v / max_white_sq));
    Vector3f mapped = numerator.cwiseQuotient(Vector3f::Ones() + v);
    return mapped;
}

void PathTracer::toneMap(QRgb *imageData, std::vector<Vector3f> &intensityValues) {
    float max_white_l = 0.0f;
    for (const auto& v : intensityValues)
    {
        float l = luminance(v);
        max_white_l = std::max(max_white_l, l);
    }

    if (max_white_l <= 0.0f)
        max_white_l = 1.0f;

    for (int i = 0; i < m_width * m_height; ++i)
    {
        Vector3f v = intensityValues[i];
        float l_old = luminance(v);

        float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
        float l_new = numerator / (1.0f + l_old);
        Vector3f mapped = change_luminance(v, l_new);

        mapped = mapped.array().pow(1.0f / 2.2f);

        int r = int(std::clamp(mapped.x(), 0.0f, 1.0f) * 255.0f);
        int g = int(std::clamp(mapped.y(), 0.0f, 1.0f) * 255.0f);
        int b = int(std::clamp(mapped.z(), 0.0f, 1.0f) * 255.0f);

        imageData[i] = qRgb(r, g, b);
    }
}
