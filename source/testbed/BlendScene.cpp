


#include <cstdint>
#include <math/vector.h>

#include "BlendScene.h"

#include <ctime>
#include <iostream>

using math::float2;
using math::float3;

namespace
{
    namespace Triangle
    {
        const float positions[] = {
            -1.0f, -1.0f,
            1.0f, -1.0f,
            0.0f,  1.0f,
            -1.0f, -1.0f,
            1.0f, 0.0f,
            -1.0f,  1.0f,
            1.0f, -1.0f,
            1.0f,  1.0f,
            -1.0f, 0.0f
        };

        const float normals[] = {
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f,
            0.0f,  0.0f, -1.0f
        };

        const float colors[] = {
            0.0f,  1.0f, 0.0f,
            0.0f,  1.0f, 0.0f,
            0.0f,  1.0f, 0.0f,
            1.0f,  0.0f, 0.0f,
            1.0f,  0.0f, 0.0f,
            1.0f,  0.0f, 0.0f,
            0.0f,  1.0f, 0.0f,
            0.0f,  1.0f, 0.0f,
            0.0f,  1.0f, 0.0f
        };

        const std::uint32_t indices[] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8
        };
    }
}

BlendScene::BlendScene()
{
}

void BlendScene::switchRenderer(Renderer* renderer)
{
    if (renderer)
    {
        material.reset(renderer->createLitMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));
        geometry.reset(renderer->create2DTriangles(&Triangle::positions[0], &Triangle::normals[0], &Triangle::colors[0], sizeof(Triangle::indices)/sizeof(uint32_t)));
    }
    else
    {
        geometry.reset();
        material.reset();
    }
}

math::float3 opposite_color(const math::float3& src, const math::float3& grey_src, const math::float3& comp, const math::float3& grey_comp)
{
    math::float3 vec = normalize(grey_src - src);

    math::float3 comp_vec = comp - grey_comp;

    float vec_len = sqrt(dot(comp_vec, comp_vec));

    return grey_comp + vec_len * vec;
}


math::float3 greypoint(const math::float3& val)
{
    const math::float3x3 toXYZ =
    {
        0.4124f, 0.3576f, 0.1805f,
        0.2126f, 0.7152f, 0.0722f,
        0.0193f, 0.1192f, 0.9505f
    };

    math::float3 xyz = (toXYZ * val);

    float Xn = 0.95047f;
    float Zn = 1.08883f;

    xyz.x = Xn*xyz.y;
    xyz.z = Zn*xyz.y;

    const math::float3x3 toRGB =
    {
        3.2406f, -1.5372f, -0.4986f,
        -0.9689f, 1.8758f, 0.0415f,
        0.0557f, -0.2040f, 1.0570f
    };

    return toRGB * xyz;
}


float hue(const math::float3& val)
{
    return atan2(sqrt(3.f) * (val.y - val.z), 2.f * val.x - val.y - val.z);
}


bool eq_hue(const math::float3& v1, const math::float3& grey_1, const math::float3& v2, const math::float3& grey_2, const float epsilon)
{
    math::float3 vec1 = normalize(v1 - grey_1);
    math::float3 vec2 = normalize(v2 - grey_2);

    return dot(vec1, vec2) > (1.f - epsilon);
}


math::float4 blend(const math::float4& src, const math::float4& dest, float time)
{
    //float a_c1 = src.w;
    //float a_c2 = 1.f - src.w;

    //classic
    //return 0.5f * src + 0.5f * dest;

    //float a_c1 = 0.2f;
    //float a_c2 = 0.8f;

    float alpha = sin(0.2f*time);
    alpha = alpha*alpha;

    //math::float3 c1 = src.xyz();
    //math::float3 c2 = dest.xyz();

    math::float3 c1 = alpha * src.xyz();
    math::float3 c2 = (1.f - alpha) * dest.xyz();

    math::float3 cnew;

    math::float3 grey_c1 = greypoint(c1);
    math::float3 grey_c2 = greypoint(c2);

    float epsilon = 0.01f;

    if (eq_hue(c1, grey_c1, c2, grey_c2, epsilon))
    {
        cnew = c1 + c2;
    }
    else
    {
        math::float3 c2_ = opposite_color(c1, grey_c1, c2, grey_c2);

        cnew = c1 + c2_;
        math::float3 grey_cnew = greypoint(cnew);

        if (!eq_hue(c1, grey_c1, cnew, grey_cnew, epsilon))
        {
            math::float3 c1_ = opposite_color(c2, grey_c2, c1, grey_c1);

            cnew = c1_ + c2;
        }
    }

    return math::float4(cnew, 1.f);
}

void BlendScene::draw(RendereringContext* context) const
{
    float time = clock() * 0.005f;
    math::float4 res = blend(math::float4(0, 1, 0, 1), math::float4(0.6f, 0.7f, 1.0f, 1.0f), time);

    std::cout << res.x << " " << res.y << " " << res.z << ";" << std::endl;

    if (res.x > res.y && res.x > res.z)
    {
        std::cout << "wat";
    }

    math::float4 toeff = blend(math::float4(0, 1, 0, 1), math::float4(0.6f, 0.7f, 1.0f, 1.0f), time);

    context->setUniformf(0, time);

    context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f));
    material->draw(geometry.get());
}
