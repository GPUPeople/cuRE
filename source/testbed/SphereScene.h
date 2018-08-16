
#ifndef INCLUDED_SPHERESCENE
#define INCLUDED_SPHERESCENE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "Resource.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"

#include "resource_ptr.h"


class SphereScene : public Scene
{
private:
	resource_ptr<Geometry> geometry;
	resource_ptr<Material> material;

public:
	SphereScene();
	SphereScene(const SphereScene&) = delete;
	SphereScene& operator =(const SphereScene&) = delete;

	void switchRenderer(Renderer* renderer);
	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif  // INCLUDED_SCENE
