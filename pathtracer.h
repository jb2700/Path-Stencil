#ifndef PATHTRACER_H
#define PATHTRACER_H

#include <QImage>

#include "scene/scene.h"

#include <Eigen/Dense>

#include <util/Common.h>

using namespace Eigen;


struct Settings {
    int samplesPerPixel;
    bool directLightingOnly; // if true, ignore indirect lighting
    int numDirectLightingSamples; // number of shadow rays to trace from each intersection point
    float pathContinuationProb; // probability of spawning a new secondary ray == (1-pathTerminationProb)
};

struct PixelGeometry {
    Vector3f color; // this is the output of trace_ray
    Vector3f normal;
    float depth;
    Vector3f albedo; // diffuse lighting
};

struct RenderResult {
    Vector3f radiance;
    Vector3f normal;
    float depth;
    Vector3f albedo;
};

class PathTracer
{
public:
    PathTracer(int width, int height);

    void traceScene(QRgb *imageData, const Scene &scene);

    const Vector3f calculateDirectLighting(const IntersectionInfo& p, const Vector3f& N,
                                                       const Vector3f& p_d, const Vector3f& p_s,
                                           const Scene& scene, const Ray& r, tinyobj::material_t mat, const Triangle *t);

    Settings settings;

private:
    int m_width, m_height;

    void toneMap(QRgb *imageData, std::vector<Eigen::Vector3f> &intensityValues);

    RenderResult tracePixel(float x, float y, const Scene &scene, const Eigen::Matrix4f &invViewMatrix);
    Eigen::Vector3f traceRay(const Ray& r, const Scene &scene, bool count_emitted);
};

#endif // PATHTRACER_H
