


#ifndef INCLUDED_GLYPH_SCENE
#define INCLUDED_GLYPH_SCENE

#pragma once

#include <memory>
#include <vector>
#include <map>

#include "Resource.h"
#include "Scene.h"

#include "resource_ptr.h"


class GlyphScene : public Scene
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
		math::float3 uvsign;
		math::float4 color;
	};

private:
	resource_ptr<Geometry> geometry;
	resource_ptr<Material> material;

	std::vector<math::uint3> indices_;

	std::vector<Vertex> final_vertices_;

	void attachOBJ(const char* filename, math::float4 color, math::float4 offscale, float rot);

public:
	GlyphScene();

	GlyphScene(const GlyphScene&) = delete;
	GlyphScene& operator =(const GlyphScene&) = delete;


	void switchRenderer(Renderer* renderer);
	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};


#endif  // INCLUDED_INDEXED_TRIANGLE_MESH
