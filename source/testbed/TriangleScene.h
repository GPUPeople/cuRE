


#ifndef INCLUDED_TRIANGLESCENE
#define INCLUDED_TRIANGLESCENE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "Resource.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"

#include "resource_ptr.h"


class TriangleScene : public Scene
{
	resource_ptr<Geometry> geometry;
	resource_ptr<Material> material;

public:
	TriangleScene() = default;

	TriangleScene(const TriangleScene&) = delete;
	TriangleScene& operator =(const TriangleScene&) = delete;

	void switchRenderer(Renderer* renderer);
	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif  // INCLUDED_TRIANGLESCENE
