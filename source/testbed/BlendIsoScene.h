


#ifndef INCLUDED_BLEND_ISO_SCENE
#define INCLUDED_BLEND_ISO_SCENE

#pragma once

#include "Scene.h"
#include "Resource.h"
#include "resource_ptr.h"
#include <memory>
#include <vector>
#include <map>

class BlendIsoScene : public Scene
{
public:

    struct cmpuint3 
    {
        bool operator()(const math::uint3& a, const math::uint3& b) const 
        {
            if (a.x == b.x)
            {
                if (a.y == b.y)
                {
                    return a.z < b.z;
                }
                return a.y < b.y;
            }
            return a.x < b.x;
        }
    };

    class Vertex
    {
    public:
        math::float3 position;
        math::float3 normal;
        math::float4 color;
    };

private:

    resource_ptr<Geometry> geometry;
    resource_ptr<Material> material;

    std::vector<math::uint3> indices_;

    std::vector<Vertex> final_vertices_;

    void attachOBJ(const char* filename, math::float4 color);

public:
    BlendIsoScene(const BlendIsoScene&) = delete;
    BlendIsoScene& operator =(const BlendIsoScene&) = delete;

    BlendIsoScene();

    void switchRenderer(Renderer* renderer);
    void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};


#endif  // INCLUDED_INDEXED_TRIANGLE_MESH
