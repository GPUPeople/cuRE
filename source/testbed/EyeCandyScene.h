


#ifndef INCLUDED_EYECANDYSCENE
#define INCLUDED_EYECANDYSCENE

#pragma once

#include <memory>

#include <math/vector.h>

#include "Resource.h"
#include "Scene.h"

#include "resource_ptr.h"


struct Renderer;
struct RendereringContext;
class Display;


class EyeCandyScene : public Scene
{
public:
	enum class ShaderType
	{
		NORMAL,
		VERTEX_HEAVY,
		FRAGMENT_HEAVY
	};

private:
	struct GPUVertex
	{
		math::float4 pos;
		math::float4 normal;
		math::float4 color;
	};

	uint32_t width, height;

	uint32_t num_vertices;
	uint32_t num_meshes;
	uint32_t num_triangles;
	std::unique_ptr<GPUVertex[]> vertices;
	std::unique_ptr<math::uint3[]> triangles;
	std::unique_ptr<math::float3[]> triangle_colors;

	resource_ptr<Material> material;
	resource_ptr<Geometry> geometry;

	bool use_drawcalls_ = false;

	struct Range
	{
		uint32_t from;
		uint32_t count;
	};

	std::vector<Range> ranges_;

	ShaderType shader_type;

public:
	EyeCandyScene(const char* scene, const Config& config, Display& display, ShaderType shader_type = ShaderType::NORMAL);

	EyeCandyScene(const EyeCandyScene&) = delete;
	EyeCandyScene& operator=(const EyeCandyScene&) = delete;

	void switchRenderer(Renderer* renderer);

	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif // INCLUDED_CLIPSPACESCENE
