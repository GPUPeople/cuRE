


#ifndef INCLUDED_ICOSAHEDRONSCENE
#define INCLUDED_ICOSAHEDRONSCENE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "Resource.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"

#include "resource_ptr.h"


class IcosahedronScene : public Scene
{
	resource_ptr<Geometry> geometry;
	resource_ptr<Texture> texture;
	resource_ptr<Material> material1;
	resource_ptr<Material> material2;
	resource_ptr<Material> material3;
	resource_ptr<Material> material4;

public:
	IcosahedronScene() = default;

	IcosahedronScene(const IcosahedronScene&) = delete;
	IcosahedronScene& operator =(const IcosahedronScene&) = delete;

	void switchRenderer(Renderer* renderer);
	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif  // INCLUDED_ICOSAHEDRONSCENE
