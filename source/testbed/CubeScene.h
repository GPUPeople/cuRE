


#ifndef INCLUDED_CUBESCENE
#define INCLUDED_CUBESCENE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "Resource.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"

#include "resource_ptr.h"


class CubeScene : public Scene
{
	resource_ptr<Geometry> geometry;
	resource_ptr<Texture> texture;
	resource_ptr<Material> material;

public:
	CubeScene() = default;

	CubeScene(const CubeScene&) = delete;
	CubeScene& operator =(const CubeScene&) = delete;

	void switchRenderer(Renderer* renderer);
	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif  // INCLUDED_CUBESCENE
