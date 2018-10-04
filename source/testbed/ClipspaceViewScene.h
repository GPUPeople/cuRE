


#ifndef INCLUDED_CLIPSPACEVIEWSCENE
#define INCLUDED_CLIPSPACEVIEWSCENE

#pragma once

#include <cstdint>
#include <memory>

#include <math/vector.h>

#include "Resource.h"
#include "Scene.h"

#include "resource_ptr.h"


struct Renderer;
struct RendereringContext;


class ClipspaceViewScene : public Scene
{
	size_t num_vertices;
	std::unique_ptr<float[]> vertices;
	std::unique_ptr<float[]> normals;
	std::unique_ptr<float[]> texcoords;

	std::unique_ptr<std::uint32_t[]> indices;

	resource_ptr<Material> material;
	resource_ptr<Geometry> geometry;

public:
	ClipspaceViewScene(const char* scenefile);

	ClipspaceViewScene(const ClipspaceViewScene&) = delete;
	ClipspaceViewScene& operator =(const ClipspaceViewScene&) = delete;

	void switchRenderer(Renderer* renderer);

	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif  // INCLUDED_CLIPSPACEVIEWSCENE
