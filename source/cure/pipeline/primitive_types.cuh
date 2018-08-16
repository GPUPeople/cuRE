


#ifndef INCLUDED_PRIMITIVE_TYPE
#define INCLUDED_PRIMITIVE_TYPE

#pragma once

#include "geometry_stage.cuh"


struct PrimitiveType
{
	template <typename Vertex>
	__device__
	static math::float4 position(unsigned int triangle, unsigned int vi, const Vertex& vertex)
	{
		return shfl(vertex.position, vi);
	}
};


struct TriangleList : PrimitiveType
{
	__device__
	static unsigned int vertexIndex(unsigned int i)
	{
		return i;
	}

	__device__
	static constexpr unsigned int maxBatchSize(unsigned int max_num_triangles)
	{
		return max_num_triangles * 3U;
	}

	__device__
	static constexpr unsigned int indices(unsigned int num_primitives)
	{
		return num_primitives * 3U;
	}

	__device__
	static constexpr unsigned int primitives(unsigned int indices)
	{
		return indices / 3U;
	}

	__device__
	static constexpr unsigned int primitiveID(unsigned int index)
	{
		return index / 3U;
	}

	__device__
	static constexpr unsigned int triangles(unsigned int indices)
	{
		return indices / 3U;
	}

	template <typename V>
	__device__
	static uint3 vertices(unsigned int triangle, V v)
	{
		auto v0 = 3U * triangle;
		return{ v0, v0 + 1, v0 + 2 };
	}
};


struct IndexedTriangleList : PrimitiveType
{
	__device__
	static unsigned int vertexIndex(unsigned int i)
	{
		return index_buffer[i];
	}

	__device__
	static constexpr unsigned int maxBatchSize(unsigned int max_num_triangles)
	{
		return max_num_triangles * 3U;
	}

	__device__
	static constexpr unsigned int indices(unsigned int num_primitives)
	{
		return num_primitives * 3U;
	}

	__device__
	static constexpr unsigned int primitives(unsigned int indices)
	{
		return indices / 3U;
	}

	__device__
	static constexpr unsigned int primitiveID(unsigned int index)
	{
		return index / 3U;
	}

	__device__
	static constexpr unsigned int triangles(unsigned int indices)
	{
		return indices / 3U;
	}

	template <typename V>
	__device__
	static uint3 vertices(unsigned int triangle, V v)
	{
		auto v0 = 3U * triangle;
		return { v0, v0 + 1, v0 + 2 };
	}
};


//struct IndexedStereoTriangleList : PrimitiveType
//{
//	__device__
//	static unsigned int vertexIndex(unsigned int i)
//	{
//		return index_buffer[i];
//	}
//
//	__device__
//	static constexpr unsigned int maxBatchSize(unsigned int max_num_triangles)
//	{
//		return max_num_triangles * 3U / 2;
//	}
//
//	__device__
//	static constexpr unsigned int indices(unsigned int num_primitives)
//	{
//		return num_primitives * 3U;
//	}
//
//	__device__
//	static constexpr unsigned int primitives(unsigned int indices)
//	{
//		return indices / 3U;
//	}
//
//	__device__
//	static constexpr unsigned int primitiveID(unsigned int index)
//	{
//		return index / 3U;
//	}
//
//	__device__
//	static constexpr unsigned int triangles(unsigned int indices)
//	{
//		return indices / 3U * 2;
//	}
//
//	template <typename V>
//	__device__
//	static uint3 vertices(unsigned int triangle, V v)
//	{
//		auto v0 = triangle / 2 * 3;
//		return { v0, v0 + 1, v0 + 2 };
//	}
//
//	template <typename Vertex>
//	__device__
//	static math::float4 position(unsigned int triangle, unsigned int vi, const Vertex& vertex)
//	{
//		auto l = shfl(vertex.position_l, vi);
//		auto r = shfl(vertex.position_r, vi);
//		return triangle & 0x1U == 0U ? l : r;
//	}
//};


struct IndexedQuadList : PrimitiveType
{
	__device__
	static unsigned int vertexIndex(unsigned int i)
	{
		return index_buffer[i];
	}

	__device__
	static constexpr unsigned int maxBatchSize(unsigned int max_num_triangles)
	{
		return max_num_triangles * 2U;
	}

	__device__
	static constexpr unsigned int indices(unsigned int num_primitives)
	{
		return num_primitives * 4U;
	}

	__device__
	static constexpr unsigned int primitives(unsigned int indices)
	{
		return indices / 4U;
	}

	__device__
	static constexpr unsigned int primitiveID(unsigned int index)
	{
		return index / 4U;
	}

	__device__
	static constexpr unsigned int triangles(unsigned int indices)
	{
		return indices / 4U * 2U;
	}

	template <typename V>
	__device__
	static uint3 vertices(unsigned int triangle, V v)
	{
		auto t = triangle & 0x1U;
		auto v0 = (triangle / 2) * 4;

		auto f = 0U;

		return{
			v0 + (((f & t) << 1U) | (~(f ^ t) & 0x1U)),
			v0 + (2U >> f),
			v0 + (((f | t) << 1U) | (f ^ t))
		};
	}
};


template <typename TriangulationShader>
struct IndexedAdaptiveQuadList : PrimitiveType
{
	__device__
	static unsigned int vertexIndex(unsigned int i)
	{
		return index_buffer[i];
	}

	__device__
	static constexpr unsigned int maxBatchSize(unsigned int max_num_triangles)
	{
		return max_num_triangles * 2U;
	}

	__device__
	static constexpr unsigned int indices(unsigned int num_primitives)
	{
		return num_primitives * 4U;
	}

	__device__
	static constexpr unsigned int primitives(unsigned int indices)
	{
		return indices / 4U;
	}

	__device__
	static constexpr unsigned int primitiveID(unsigned int index)
	{
		return index / 4U;
	}

	__device__
	static constexpr unsigned int triangles(unsigned int indices)
	{
		return indices / 4U * 2U;
	}

	template <typename V>
	__device__
	static uint3 vertices(unsigned int triangle, V v)
	{
		auto t = triangle & 0x1U;
		auto i0 = (triangle / 2) * 4;

		//auto v1 = v(i0 + t);
		//auto v2 = v(i0 + (2U | t));
		auto v1 = v(i0 + t);
		auto v2 = v(i0 + t + 1);
		auto v3 = v(i0 + t + 2);
		auto v4 = v((i0 + t + 3) & 0x3U);
		TriangulationShader shader {};
		float d = shader(v1, v2, v3, v4);
		//float d = shader(v1, v2);
		auto f = __shfl_down_sync(~0U, d, 1U, 2U) > __shfl_up_sync(~0U, d, 1U, 2U) ? 0U : 1U;


		//  f=0   f=1
		//
		//  +-+   +-+ 
		//  |/|   |\|
		//  +-+   +-+
		//
		//  f  t   1   2   3
		//--------------------
		//  0  0  01  10  00
		//  0  1  00  10  11
		//  1  0  00  01  11
		//  1  1  11  01  10

		return {
			i0 + (((f & t) << 1U) | (~(f ^ t) & 0x1U)),
			i0 + (2U >> f),
			i0 + (((f | t) << 1U) |   (f ^ t))
		};
	}
};


#endif  // INCLUDED_PRIMITIVE_TYPE
