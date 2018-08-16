


#ifndef INCLUDED_COVERAGESCENE
#define INCLUDED_COVERAGESCENE

#pragma once

#include <memory>

#include <math/vector.h>

#include "Resource.h"

#include "Scene.h"
#include "resource_ptr.h"


struct Renderer;
struct RendereringContext;
class Display;


class CheckerboardScene : public Scene
{
private:
	struct GPUVertex
	{
		math::float4 pos;
		math::float4 normal;
		math::float4 color;
	};

	 uint32_t width;
	 uint32_t height;

	uint32_t num_vertices;
	uint32_t num_meshes;
	uint32_t num_triangles;
	std::unique_ptr<GPUVertex[]> vertices;
	std::unique_ptr<math::uint3[]> triangles;
	std::unique_ptr<math::float3[]> triangle_colors;

	resource_ptr<Material> material;
	resource_ptr<Geometry> geometry;

	unsigned int type;

public:
	CheckerboardScene(const Config& config, Display& display);

	CheckerboardScene(const CheckerboardScene&) = delete;
	CheckerboardScene& operator=(const CheckerboardScene&) = delete;

	void switchRenderer(Renderer* renderer);

	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif // INCLUDED_CLIPSPACESCENE
