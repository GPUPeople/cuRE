


#ifndef INCLUDED_CLIPSPACESCENE
#define INCLUDED_CLIPSPACESCENE

#pragma once

#include <memory>

#include <math/vector.h>

#include "Resource.h"
#include "Scene.h"

#include "resource_ptr.h"

struct Renderer;
struct RendereringContext;


class ClipspaceScene : public Scene
{
public:
	enum class ShaderType
	{
		NORMAL,
		VERTEX_HEAVY,
		VERTEX_SUPER_HEAVY,
		FRAGMENT_HEAVY,
		FRAGMENT_SUPER_HEAVY
	};

private:
	size_t num_vertices;
	std::unique_ptr<float[]> vertices;

	resource_ptr<Material> material;
	resource_ptr<Geometry> geometry;

	ShaderType shader_type;

public:
	ClipspaceScene(const char* scenefile, ShaderType shader_type = ShaderType::NORMAL);

	ClipspaceScene(const ClipspaceScene&) = delete;
	ClipspaceScene& operator =(const ClipspaceScene&) = delete;

	void switchRenderer(Renderer* renderer);

	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif  // INCLUDED_CLIPSPACESCENE
