


#include <cstdint>
#include <math/vector.h>

#include "IndexedTriangleMesh.h"


using math::float2;
using math::float3;

namespace
{
	struct Vertex
	{
		float3 p;
		float3 n;

		Vertex(const float3& p, const float3& n)
			: p(p), n(n)
		{
		}
	};

	const float phi = 1.6180339887498948482045868343656f;
	const float iphi = 1.0f / phi;

	const float2 n = normalize(float2(1.0f, phi - 1.0f));

	const Vertex vertices[] = {
		Vertex(float3(-1.0f, 1.0f,-1.0f), float3(-n.y, 0.0f, n.x)),
		Vertex(float3(-phi, 0.0f,-iphi), float3(-n.y, 0.0f, n.x)),
		Vertex(float3(-1.0f, -1.0f,-1.0f), float3(-n.y, 0.0f, n.x)),
		Vertex(float3(0.0f, -iphi,-phi), float3(-n.y, 0.0f, n.x)),
		Vertex(float3(0.0f, iphi,-phi), float3(-n.y, 0.0f, n.x)),

		Vertex(float3(1.0f, 1.0f,-1.0f), float3(n.y, 0.0f, n.x)),
		Vertex(float3(0.0f, iphi,-phi), float3(n.y, 0.0f, n.x)),
		Vertex(float3(0.0f, -iphi,-phi), float3(n.y, 0.0f, n.x)),
		Vertex(float3(1.0f, -1.0f,-1.0f), float3(n.y, 0.0f, n.x)),
		Vertex(float3(phi, 0.0f,-iphi), float3(n.y, 0.0f, n.x)),

		Vertex(float3(1.0f, 1.0f,-1.0f), float3(n.x, n.y, 0.0f)),
		Vertex(float3(phi, 0.0f,-iphi), float3(n.x, n.y, 0.0f)),
		Vertex(float3(phi, 0.0f, iphi), float3(n.x, n.y, 0.0f)),
		Vertex(float3(1.0f, 1.0f, 1.0f), float3(n.x, n.y, 0.0f)),
		Vertex(float3(iphi, phi, 0.0f), float3(n.x, n.y, 0.0f)),

		Vertex(float3(1.0f, 1.0f, 1.0f), float3(0.0f, n.x, -n.y)),
		Vertex(float3(0.0f, iphi, phi), float3(0.0f, n.x, -n.y)),
		Vertex(float3(-1.0f, 1.0f, 1.0f), float3(0.0f, n.x, -n.y)),
		Vertex(float3(-iphi, phi, 0.0f), float3(0.0f, n.x, -n.y)),
		Vertex(float3(iphi, phi, 0.0f), float3(0.0f, n.x, -n.y)),

		Vertex(float3(-1.0f, 1.0f, 1.0f), float3(-n.x, n.y, 0.0f)),
		Vertex(float3(-phi, 0.0f, iphi), float3(-n.x, n.y, 0.0f)),
		Vertex(float3(-phi, 0.0f,-iphi), float3(-n.x, n.y, 0.0f)),
		Vertex(float3(-1.0f, 1.0f, 1.0f), float3(-n.x, n.y, 0.0f)),
		Vertex(float3(-iphi, phi, 0.0f), float3(-n.x, n.y, 0.0f)),

		Vertex(float3(-1.0f, 1.0f,-1.0f), float3(0.0f, n.x, n.y)),
		Vertex(float3(0.0f, iphi,-phi), float3(0.0f, n.x, n.y)),
		Vertex(float3(1.0f, 1.0f,-1.0f), float3(0.0f, n.x, n.y)),
		Vertex(float3(iphi, phi, 0.0f), float3(0.0f, n.x, n.y)),
		Vertex(float3(-iphi, phi, 0.0f), float3(0.0f, n.x, n.y)),


		Vertex(float3(-1.0f, -1.0f,-1.0f), float3(0.0f, -n.x, n.y)),
		Vertex(float3(-iphi, -phi, 0.0f), float3(0.0f, -n.x, n.y)),
		Vertex(float3(iphi, -phi, 0.0f), float3(0.0f, -n.x, n.y)),
		Vertex(float3(1.0f, -1.0f,-1.0f), float3(0.0f, -n.x, n.y)),
		Vertex(float3(0.0f, -iphi,-phi), float3(0.0f, -n.x, n.y)),

		Vertex(float3(1.0f, -1.0f,-1.0f), float3(n.x, -n.y, 0.0f)),
		Vertex(float3(iphi, -phi, 0.0f), float3(n.x, -n.y, 0.0f)),
		Vertex(float3(1.0f, -1.0f, 1.0f), float3(n.x, -n.y, 0.0f)),
		Vertex(float3(phi, 0.0f, iphi), float3(n.x, -n.y, 0.0f)),
		Vertex(float3(phi, 0.0f,-iphi), float3(n.x, -n.y, 0.0f)),

		Vertex(float3(1.0f, -1.0f, 1.0f), float3(n.y, 0.0f, -n.x)),
		Vertex(float3(0.0f, -iphi, phi), float3(n.y, 0.0f, -n.x)),
		Vertex(float3(0.0f, iphi, phi), float3(n.y, 0.0f, -n.x)),
		Vertex(float3(1.0f, 1.0f, 1.0f), float3(n.y, 0.0f, -n.x)),
		Vertex(float3(phi, 0.0f, iphi), float3(n.y, 0.0f, -n.x)),

		Vertex(float3(-1.0f, -1.0f, 1.0f), float3(-n.y, 0.0f, -n.x)),
		Vertex(float3(-phi, 0.0f, iphi), float3(-n.y, 0.0f, -n.x)),
		Vertex(float3(-1.0f, 1.0f, 1.0f), float3(-n.y, 0.0f, -n.x)),
		Vertex(float3(0.0f, iphi, phi), float3(-n.y, 0.0f, -n.x)),
		Vertex(float3(0.0f, -iphi, phi), float3(-n.y, 0.0f, -n.x)),

		Vertex(float3(-1.0f, -1.0f, 1.0f), float3(-n.x, -n.y, 0.0f)),
		Vertex(float3(-iphi, -phi, 0.0f), float3(-n.x, -n.y, 0.0f)),
		Vertex(float3(-1.0f, -1.0f,-1.0f), float3(-n.x, -n.y, 0.0f)),
		Vertex(float3(-phi, 0.0f,-iphi), float3(-n.x, -n.y, 0.0f)),
		Vertex(float3(-phi, 0.0f, iphi), float3(-n.x, -n.y, 0.0f)),

		Vertex(float3(1.0f, -1.0f, 1.0f), float3(0.0f, -n.x, -n.y)),
		Vertex(float3(iphi, -phi, 0.0f), float3(0.0f, -n.x, -n.y)),
		Vertex(float3(-iphi, -phi, 0.0f), float3(0.0f, -n.x, -n.y)),
		Vertex(float3(-1.0f, -1.0f, 1.0f), float3(0.0f, -n.x, -n.y)),
		Vertex(float3(0.0f, -iphi, phi), float3(0.0f, -n.x, -n.y)),
	};

	const std::uint32_t indices[] = {
		1, 2, 0, 2, 3, 0, 3, 4, 0,
		6, 7, 5, 7, 8, 5, 8, 9, 5,
		11, 12, 10, 12, 13, 10, 13, 14, 10,
		16, 17, 15, 17, 18, 15, 18, 19, 15,
		21, 22, 20, 22, 23, 20, 23, 24, 20,
		26, 27, 25, 27, 28, 25, 28, 29, 25,
		31, 32, 30, 32, 33, 30, 33, 34, 30,
		36, 37, 35, 37, 38, 35, 38, 39, 35,
		41, 42, 40, 42, 43, 40, 43, 44, 40,
		46, 47, 45, 47, 48, 45, 48, 49, 45,
		51, 52, 50, 52, 53, 50, 53, 54, 50,
		56, 57, 55, 57, 58, 55, 58, 59, 55
	};
}

IndexedTrianglemesh::IndexedTrianglemesh()
{
}

