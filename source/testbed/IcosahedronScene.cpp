


#include <cstdint>
#include <type_traits>
#include <stdexcept>

#include <vector>
#include <map>
#include <set>

#include <math/vector.h>

#include "IcosahedronScene.h"

using math::float2;
using math::float3;


namespace
{
	const float phi = 1.6180339887498948482045868343656f;
	const float iphi = 1.0f / phi;


	const float2 n = normalize(float2(1.0f, phi - 1.0f));

	//const float3 positions[] = { { 0, -1, -0.5 }, { 1, -1, -0.5 }, { 0.5, 0, -0.5 }, { -1, -1, -0.5 }, { -0.5, 0, -0.5 }, { 0, 1, -0.5 } };
	const float3 positions[] = {
		float3(-1.0f, 1.0f, -1.0f),
		float3(-phi, 0.0f, -iphi),
		float3(-1.0f, -1.0f, -1.0f),
		float3(0.0f, -iphi, -phi),
		float3(0.0f, iphi, -phi),

		float3(1.0f, 1.0f, -1.0f),
		float3(0.0f, iphi, -phi),
		float3(0.0f, -iphi, -phi),
		float3(1.0f, -1.0f, -1.0f),
		float3(phi, 0.0f, -iphi),

		float3(1.0f, 1.0f, -1.0f),
		float3(phi, 0.0f, -iphi),
		float3(phi, 0.0f, iphi),
		float3(1.0f, 1.0f, 1.0f),
		float3(iphi, phi, 0.0f),

		float3(1.0f, 1.0f, 1.0f),
		float3(0.0f, iphi, phi),
		float3(-1.0f, 1.0f, 1.0f),
		float3(-iphi, phi, 0.0f),
		float3(iphi, phi, 0.0f),

		float3(-1.0f, 1.0f, 1.0f),
		float3(-phi, 0.0f, iphi),
		float3(-phi, 0.0f, -iphi),
		float3(-1.0f, 1.0f, -1.0f),
		float3(-iphi, phi, 0.0f),

		float3(-1.0f, 1.0f, -1.0f),
		float3(0.0f, iphi, -phi),
		float3(1.0f, 1.0f, -1.0f),
		float3(iphi, phi, 0.0f),
		float3(-iphi, phi, 0.0f),


		float3(-1.0f, -1.0f, -1.0f),
		float3(-iphi, -phi, 0.0f),
		float3(iphi, -phi, 0.0f),
		float3(1.0f, -1.0f, -1.0f),
		float3(0.0f, -iphi, -phi),

		float3(1.0f, -1.0f, -1.0f),
		float3(iphi, -phi, 0.0f),
		float3(1.0f, -1.0f, 1.0f),
		float3(phi, 0.0f, iphi),
		float3(phi, 0.0f, -iphi),

		float3(1.0f, -1.0f, 1.0f),
		float3(0.0f, -iphi, phi),
		float3(0.0f, iphi, phi),
		float3(1.0f, 1.0f, 1.0f),
		float3(phi, 0.0f, iphi),

		float3(-1.0f, -1.0f, 1.0f),
		float3(-phi, 0.0f, iphi),
		float3(-1.0f, 1.0f, 1.0f),
		float3(0.0f, iphi, phi),
		float3(0.0f, -iphi, phi),

		float3(-1.0f, -1.0f, 1.0f),
		float3(-iphi, -phi, 0.0f),
		float3(-1.0f, -1.0f, -1.0f),
		float3(-phi, 0.0f, -iphi),
		float3(-phi, 0.0f, iphi),

		float3(1.0f, -1.0f, 1.0f),
		float3(iphi, -phi, 0.0f),
		float3(-iphi, -phi, 0.0f),
		float3(-1.0f, -1.0f, 1.0f),
		float3(0.0f, -iphi, phi)
	};

	const float3 normals[] = {
		float3(-n.y, 0.0f, -n.x),
		float3(-n.y, 0.0f, -n.x),
		float3(-n.y, 0.0f, -n.x),
		float3(-n.y, 0.0f, -n.x),
		float3(-n.y, 0.0f, -n.x),

		float3(n.y, 0.0f, -n.x),
		float3(n.y, 0.0f, -n.x),
		float3(n.y, 0.0f, -n.x),
		float3(n.y, 0.0f, -n.x),
		float3(n.y, 0.0f, -n.x),

		float3(n.x, n.y, 0.0f),
		float3(n.x, n.y, 0.0f),
		float3(n.x, n.y, 0.0f),
		float3(n.x, n.y, 0.0f),
		float3(n.x, n.y, 0.0f),

		float3(0.0f, n.x, n.y),
		float3(0.0f, n.x, n.y),
		float3(0.0f, n.x, n.y),
		float3(0.0f, n.x, n.y),
		float3(0.0f, n.x, n.y),

		float3(-n.x, n.y, 0.0f),
		float3(-n.x, n.y, 0.0f),
		float3(-n.x, n.y, 0.0f),
		float3(-n.x, n.y, 0.0f),
		float3(-n.x, n.y, 0.0f),

		float3(0.0f, n.x, -n.y),
		float3(0.0f, n.x, -n.y),
		float3(0.0f, n.x, -n.y),
		float3(0.0f, n.x, -n.y),
		float3(0.0f, n.x, -n.y),


		float3(0.0f, -n.x, -n.y),
		float3(0.0f, -n.x, -n.y),
		float3(0.0f, -n.x, -n.y),
		float3(0.0f, -n.x, -n.y),
		float3(0.0f, -n.x, -n.y),

		float3(n.x, -n.y, 0.0f),
		float3(n.x, -n.y, 0.0f),
		float3(n.x, -n.y, 0.0f),
		float3(n.x, -n.y, 0.0f),
		float3(n.x, -n.y, 0.0f),

		float3(n.y, 0.0f, n.x),
		float3(n.y, 0.0f, n.x),
		float3(n.y, 0.0f, n.x),
		float3(n.y, 0.0f, n.x),
		float3(n.y, 0.0f, n.x),

		float3(-n.y, 0.0f, n.x),
		float3(-n.y, 0.0f, n.x),
		float3(-n.y, 0.0f, n.x),
		float3(-n.y, 0.0f, n.x),
		float3(-n.y, 0.0f, n.x),

		float3(-n.x, -n.y, 0.0f),
		float3(-n.x, -n.y, 0.0f),
		float3(-n.x, -n.y, 0.0f),
		float3(-n.x, -n.y, 0.0f),
		float3(-n.x, -n.y, 0.0f),

		float3(0.0f, -n.x, n.y),
		float3(0.0f, -n.x, n.y),
		float3(0.0f, -n.x, n.y),
		float3(0.0f, -n.x, n.y),
		float3(0.0f, -n.x, n.y)
	};

	size_t num_vertices = std::extent<decltype(positions)>::value;

	//const std::uint32_t indices[] = { 0, 1, 2, 3, 0, 4, 4, 2, 5 };

	const std::uint32_t indices[] = {
		1, 0, 2, 2, 0, 3, 3, 0, 4,
		6, 5, 7, 7, 5, 8, 8, 5, 9,
		11, 10, 12, 12, 10, 13, 13, 10, 14,
		16, 15, 17, 17, 15, 18, 18, 15, 19,
		21, 20, 22, 22, 20, 23, 23, 20, 24,
		26, 25, 27, 27, 25, 28, 28, 25, 29,
		31, 30, 32, 32, 30, 33, 33, 30, 34,
		36, 35, 37, 37, 35, 38, 38, 35, 39,
		41, 40, 42, 42, 40, 43, 43, 40, 44,
		46, 45, 47, 47, 45, 48, 48, 45, 49,
		51, 50, 52, 52, 50, 53, 53, 50, 54,
		56, 55, 57, 57, 55, 58, 58, 55, 59
	};

	size_t num_indices = std::extent<decltype(indices)>::value;
}

void IcosahedronScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		material1 = resource_ptr<Material>(renderer->createColoredMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));
		material2 = resource_ptr<Material>(renderer->createLitMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));
		material3 = resource_ptr<Material>(renderer->createLitMaterial(math::float4(0.0f, 1.0f, 1.0f, 1.0f)));
		material4 = resource_ptr<Material>(renderer->createLitMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));

		geometry = resource_ptr<Geometry>(renderer->createIndexedTriangles(&positions[0].x, &normals[0].x, &positions[0].x, num_vertices, indices, num_indices));

		if (!material1 || !material2 || !material3 ||!material4 || !geometry)
			throw std::runtime_error("renderer cannot support this scene type");
	}
	else
	{
		geometry.reset();
		texture.reset();
		material1.reset();
		material2.reset();
		material3.reset();
		material4.reset();
	}
}

void IcosahedronScene::draw(RendereringContext* context) const
{
	context->setLight(math::float3(0.0f, 10.0f, 0.0f), math::float3(1.0f, 1.0f, 1.0f));

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, -2.0f,
		                                          0.0f, 1.0f, 0.0f, 0.0f,
		                                          0.0f, 0.0f, 1.0f, -2.0f));
	material1->draw(geometry.get());

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, -2.0f,
		                                          0.0f, 1.0f, 0.0f, 0.0f,
		                                          0.0f, 0.0f, 1.0f, 2.0f));
	material2->draw(geometry.get());

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 2.0f,
		                                          0.0f, 1.0f, 0.0f, 0.0f,
		                                          0.0f, 0.0f, 1.0f, 2.0f));
	material3->draw(geometry.get());

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 2.0f,
		                                          0.0f, 1.0f, 0.0f, 0.0f,
		                                          0.0f, 0.0f, 1.0f,-2.0f));
	material4->draw(geometry.get());
}
