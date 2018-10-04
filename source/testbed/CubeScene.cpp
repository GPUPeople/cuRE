


#include <cstdint>
#include <type_traits>
#include <stdexcept>

#include <vector>
#include <map>
#include <set>

#include <math/vector.h>

#include "CubeScene.h"

using math::float2;
using math::float3;


namespace
{
	float3 vertices[] = {
		{  1.0f, -1.0f, -1.0f },
		{  1.0f, -1.0f,  1.0f },
		{ -1.0f, -1.0f,  1.0f },
		{ -1.0f, -1.0f, -1.0f },

		{ -1.0f,  1.0f, -1.0f },
		{ -1.0f,  1.0f,  1.0f },
		{  1.0f,  1.0f,  1.0f },
		{  1.0f,  1.0f, -1.0f },

		{ -1.0f, -1.0f, -1.0f },
		{ -1.0f,  1.0f, -1.0f },
		{  1.0f,  1.0f, -1.0f },
		{  1.0f, -1.0f, -1.0f },

		{  1.0f, -1.0f,  1.0f },
		{  1.0f,  1.0f,  1.0f },
		{ -1.0f,  1.0f,  1.0f },
		{ -1.0f, -1.0f,  1.0f },

		{ -1.0f, -1.0f,  1.0f },
		{ -1.0f,  1.0f,  1.0f },
		{ -1.0f,  1.0f, -1.0f },
		{ -1.0f, -1.0f, -1.0f },

		{  1.0f, -1.0f, -1.0f },
		{  1.0f,  1.0f, -1.0f },
		{  1.0f,  1.0f,  1.0f },
		{  1.0f, -1.0f,  1.0f }
	};

	float3 normals[] = {
		{  0.0f, -1.0f,  0.0f },
		{  0.0f, -1.0f,  0.0f },
		{  0.0f, -1.0f,  0.0f },
		{  0.0f, -1.0f,  0.0f },

		{  0.0f,  1.0f,  0.0f },
		{  0.0f,  1.0f,  0.0f },
		{  0.0f,  1.0f,  0.0f },
		{  0.0f,  1.0f,  0.0f },

		{  0.0f,  0.0f, -1.0f },
		{  0.0f,  0.0f, -1.0f },
		{  0.0f,  0.0f, -1.0f },
		{  0.0f,  0.0f, -1.0f },

		{  0.0f,  0.0f,  1.0f },
		{  0.0f,  0.0f,  1.0f },
		{  0.0f,  0.0f,  1.0f },
		{  0.0f,  0.0f,  1.0f },

		{ -1.0f,  0.0f,  0.0f },
		{ -1.0f,  0.0f,  0.0f },
		{ -1.0f,  0.0f,  0.0f },
		{ -1.0f,  0.0f,  0.0f },

		{  1.0f,  0.0f,  0.0f },
		{  1.0f,  0.0f,  0.0f },
		{  1.0f,  0.0f,  0.0f },
		{  1.0f,  0.0f,  0.0f }
	};

	float2 texcoords[] = {
		{  0.0f,  1.0f },
		{  0.0f,  0.0f },
		{  1.0f,  0.0f },
		{  1.0f,  1.0f },

		{  0.0f,  1.0f },
		{  0.0f,  0.0f },
		{  1.0f,  0.0f },
		{  1.0f,  1.0f },

		{  0.0f,  1.0f },
		{  0.0f,  0.0f },
		{  1.0f,  0.0f },
		{  1.0f,  1.0f },

		{  0.0f,  1.0f },
		{  0.0f,  0.0f },
		{  1.0f,  0.0f },
		{  1.0f,  1.0f },

		{  0.0f,  1.0f },
		{  0.0f,  0.0f },
		{  1.0f,  0.0f },
		{  1.0f,  1.0f },

		{  0.0f,  1.0f },
		{  0.0f,  0.0f },
		{  1.0f,  0.0f },
		{  1.0f,  1.0f },
	};

	auto num_vertices = std::extent<decltype(vertices)>::value;


	std::uint32_t indices[] = {
		 0,  1,  2,  3,
		 4,  5,  6,  7,
		 8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,

		0,  1,  2,  3,
		4,  5,  6,  7,
		8,  9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23
	};

	auto num_indices = std::extent<decltype(indices)>::value;
}

void CubeScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		material = resource_ptr<Material>(renderer->createLitMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));

		geometry = resource_ptr<Geometry>(renderer->createIndexedQuads(&vertices[0].x, &normals[0].x, &texcoords[0].x, num_vertices, indices, num_indices));

		if (!material || !geometry)
			throw std::runtime_error("renderer cannot support this scene type");
	}
	else
	{
		geometry.reset();
		texture.reset();
		material.reset();
	}
}

void CubeScene::draw(RendereringContext* context) const
{
	context->setLight(math::float3(0.0f, 10.0f, 0.0f), math::float3(1.0f, 1.0f, 1.0f));

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, -2.0f,
	                                           0.0f, 1.0f, 0.0f, 0.0f,
	                                           0.0f, 0.0f, 1.0f, -2.0f));
	material->draw(geometry.get());
}
