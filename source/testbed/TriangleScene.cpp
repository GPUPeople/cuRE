


#include <cstdint>
#include <stdexcept>

#include <math/vector.h>

#include "TriangleScene.h"


namespace
{
	namespace Triangle
	{
		const float positions[] = {
			-1.0f, -1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			 0.0f,  1.0f, 0.0f
		};

		const float normals[] = {
			 0.0f,  0.0f, -1.0f,
			 0.0f,  0.0f, -1.0f,
			 0.0f,  0.0f, -1.0f
		};

		const std::uint32_t indices[] = {
			0, 1, 2
		};
	}
}

void TriangleScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		material.reset(renderer->createColoredMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));
		geometry.reset(renderer->createIndexedTriangles(&Triangle::positions[0], &Triangle::normals[0], &Triangle::positions[0], 3, &Triangle::indices[0], 3));

		if (!material || !geometry)
			throw std::runtime_error("renderer cannot support this scene type");
	}
	else
	{
		geometry.reset();
		material.reset();
	}
}

void TriangleScene::draw(RendereringContext* context) const
{
	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
	                                            0.0f, 1.0f, 0.0f, 0.0f,
	                                            0.0f, 0.0f, 1.0f, 0.0f));
	material->draw(geometry.get());
}
