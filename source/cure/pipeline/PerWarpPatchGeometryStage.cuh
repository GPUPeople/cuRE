


#ifndef INCLUDED_CURE_PER_WARP_PATCH_CACHED_GEOMETRY_STAGE
#define INCLUDED_CURE_PER_WARP_PATCH_CACHED_GEOMETRY_STAGE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include <ptx_primitives.cuh>

#include "config.h"
#include "common.cuh"

#include "instrumentation.cuh"

#include "clipping.cuh"
#include "viewport.cuh"
#include "geometry_stage.cuh"
#include "vertex_shader_stage.cuh"
#include "rasterization_stage.cuh"



template <typename T>
__device__
inline T shfl(T v, int src_lane)
{
	return __shfl_sync(~0U, v, src_lane);
}

__device__
inline math::float4 shfl(const math::float4& v, int src_lane)
{
	return math::float4(__shfl_sync(~0U, v.x, src_lane), __shfl_sync(~0U, v.y, src_lane), __shfl_sync(~0U, v.z, src_lane), __shfl_sync(~0U, v.w, src_lane));
}

__device__
inline math::float3 shfl(const math::float3& v, int src_lane)
{
	return math::float3(__shfl_sync(~0U, v.x, src_lane), __shfl_sync(~0U, v.y, src_lane), __shfl_sync(~0U, v.z, src_lane));
}


__device__
inline float snapCoordinate(float v, float w)
{
	float scale = w / 1.0f;

	return round(v / scale) * scale;
}

__device__
inline math::float4 snapVertex(const math::float4& v)
{
	return { snapCoordinate(v.x, v.w * pixel_scale.x), snapCoordinate(v.y, v.w * pixel_scale.y), v.z, v.w };
}



template <unsigned int NUM_WARPS, class InputVertexAttributes, class PrimitiveType, class VertexShader, class FragmentShader, class RasterizationStage, bool CLIPPING>
class PerWarpPatchCachedGeometryStage
{
	static constexpr int BATCH_SIZE = PrimitiveType::maxBatchSize(WARP_SIZE);

	using FragmentShaderInputs = typename ::FragmentShaderInfo<FragmentShader>::Inputs;
	using VertexShaderOutputStorage = ::VertexShaderOutputStorage<FragmentShaderInputs>;

	static_assert(VertexShaderOutputStorage::NUM_INTERPOLATORS <= NUM_INTERPOLATORS, "not enough interpolators");


	struct SharedState
	{
		unsigned int vertex_map[NUM_WARPS][static_divup<BATCH_SIZE, WARP_SIZE>::value * WARP_SIZE];
		VertexShaderOutputStorage vs_output_storage[NUM_WARPS][WARP_SIZE];
		volatile unsigned int processed_indicess[NUM_WARPS];
		volatile unsigned int current_start_indices[NUM_WARPS];
		unsigned int block_start_index;

		__device__ __forceinline__
		unsigned int start_index()
		{
			return block_start_index + warp_id() * BATCH_SIZE;
		}

		__device__ __forceinline__
		volatile unsigned int& current_start_index()
		{
			return current_start_indices[warp_id()];
		}

		__device__ __forceinline__
		volatile unsigned int& processed_indices()
		{
			return processed_indicess[warp_id()];
		}

		__device__ __forceinline__
		unsigned int end_index()
		{
			return min(start_index() + BATCH_SIZE, num_indices);
		}
		__device__ __forceinline__
		unsigned int* vertex_mapping()
		{
			return vertex_map[warp_id()];
		}
		__device__ __forceinline__
		VertexShaderOutputStorage* vertexshader_output_storage()
		{
			return vs_output_storage[warp_id()];
		}
	};


	__device__
	static void processTriangle(unsigned int primitive_id, math::float4 p1, math::float4 p2, math::float4 p3, const VertexShaderOutputStorage& o1, const VertexShaderOutputStorage& o2, const VertexShaderOutputStorage& o3)
	{
		//p1 = snapVertex(p1);
		//p2 = snapVertex(p2);
		//p3 = snapVertex(p3);

		math::float2 bounds_min;
		math::float2 bounds_max;

		if (clipTriangle(p1, p2, p3, bounds_min, bounds_max))
			return;

		if (bounds_min.x > 1.0f || bounds_max.x < -1.0f || bounds_min.y > 1.0f || bounds_max.y < -1.0f)
			return;

		math::int4 bounds = computeRasterBounds(bounds_min, bounds_max);
		if (bounds.z <= bounds.x || bounds.w <= bounds.y)
			return;


		math::float3x3 M = math::float3x3(
			p1.x, p2.x, p3.x,
			p1.y, p2.y, p3.y,
			p1.w, p2.w, p3.w
		);

		math::float3x3 M_adj = adj(M);

		float det = dot(M_adj.row1(), M.column1());

		if (BACKFACE_CULLING > 0 && det >= 0.0f)//-1e-9f)
			return;
		else if (BACKFACE_CULLING < 0 && det <= 0.0f)//1e-9f)
			return;
		else if (det == 0.0f)
			return;

		math::float3x3 M_inv = (1.0f / det) * M_adj;

		math::float3 uz = math::float3(p1.z, p2.z, p3.z) * M_inv;


		//unsigned int triangle_id = triangle_buffer.allocateWarp(NUM_BLOCKS);
		unsigned int triangle_id = triangle_buffer.allocate(RasterizationStage::MAX_TRIANGLE_REFERENCES);

		triangle_buffer.storeTriangle(triangle_id, M_inv, uz, bounds);
		store(triangle_buffer, triangle_id, o1, o2, o3);

		__threadfence();

		unsigned int num = RasterizationStage::enqueueTriangle(triangle_id, primitive_id, bounds);


		//store(triangle_buffer, triangle_id, o1, o2, o3);

		triangle_buffer.release(triangle_id, RasterizationStage::MAX_TRIANGLE_REFERENCES - num);
	}

	__device__
	static void processBatch(SharedState& state)
	{
		{
			Instrumentation::BlockObserver<2, 2> observer;
			state.current_start_index() = state.start_index();
			while (state.current_start_index() < state.end_index())
			{
				unsigned int fill_counter = 0U;
				unsigned int my_id = 0xFFFFFFFFU;
				state.processed_indices() = 0U;

				unsigned int read_offset = state.current_start_index();
				for (; read_offset < state.end_index() && fill_counter < 32U; read_offset += 32U)
				{
					unsigned int lid = laneid();

					unsigned int incoming_id = 0xFFFFFFFFU;
					unsigned int outgoing_id = 0xFFFFFFFFU;

					if (read_offset + lid < state.end_index())
						incoming_id = PrimitiveType::vertexIndex(read_offset + lid);

					#pragma unroll
					for (unsigned int i = 0; i < 32U; ++i)
					{
						unsigned int current = __shfl_sync(~0U, incoming_id, i);
						unsigned int match_mask = __ballot_sync(~0U, current == my_id);

						if (match_mask == 0)
						{
							if (fill_counter == lid)
								my_id = current;
							match_mask = 1U << fill_counter;
							++fill_counter;
						}

						if (i == lid)
							outgoing_id = match_mask;
					}

					// TODO: here be bank conflicts
					unsigned int vertex_map_id = state.processed_indices() + lid;
					state.vertex_mapping()[vertex_map_id] = __ffs(outgoing_id) - 1U;

					unsigned int newProcessedIndices = min(32U, __ffs(__ballot_sync(~0U, outgoing_id == 0U || incoming_id == 0xFFFFFFFFU)) - 1U);
					state.processed_indices() += newProcessedIndices;
				}

				unsigned int processed_vertices = min(fill_counter, 32U);
				state.processed_indices() = PrimitiveType::primitives(state.processed_indices());


				decltype(callVertexShader(VertexShader(), state.vertexshader_output_storage()[laneid()], InputVertexAttributes(my_id))) vertex;

				if (laneid() < processed_vertices)
				{
					vertex = callVertexShader(VertexShader(), state.vertexshader_output_storage()[laneid()], InputVertexAttributes(my_id));
				}
			

				{
					Instrumentation::BlockObserver<3, 2> observer;

					int tri_id = laneid();

					auto vi = PrimitiveType::vertices(tri_id, [&, vertex](int i) { return shfl(vertex, state.vertex_mapping()[i]); });

					unsigned int vi1 = state.vertex_mapping()[vi.x];
					unsigned int vi2 = state.vertex_mapping()[vi.y];
					unsigned int vi3 = state.vertex_mapping()[vi.z];

					math::float4 p1 = PrimitiveType::position(tri_id, vi1, vertex);
					math::float4 p2 = PrimitiveType::position(tri_id, vi2, vertex);
					math::float4 p3 = PrimitiveType::position(tri_id, vi3, vertex);

					if (tri_id < PrimitiveType::triangles(PrimitiveType::indices(state.processed_indices())))
					{
						unsigned int triangleId = PrimitiveType::triangles(state.current_start_index()) + tri_id;
						processTriangle(triangleId, p1, p2, p3, state.vertexshader_output_storage()[vi1], state.vertexshader_output_storage()[vi2], state.vertexshader_output_storage()[vi3]);
						__threadfence();
						tri_id = laneid();  // save a register
						triangleId = PrimitiveType::triangles(state.current_start_index()) + tri_id;
						RasterizationStage::completedPrimitive(triangleId);
					}
				}

				state.processed_indices() = PrimitiveType::indices(state.processed_indices());
				state.current_start_index() = state.current_start_index() + state.processed_indices();
			}
		}
	}

public:
	static constexpr size_t SHARED_MEMORY = sizeof(SharedState);

	__device__
	static bool run(char* shared_memory)
	{
		Instrumentation::BlockObserver<1, 1> observer;

		SharedState& shared = *reinterpret_cast<SharedState*>(shared_memory);

		if (threadIdx.x == 0)
		{
			if (index_counter < num_indices)
				shared.block_start_index = atomicAdd(&index_counter, BATCH_SIZE * NUM_WARPS);
			else
				shared.block_start_index = num_indices;
		}

		__syncthreads();

		if (shared.block_start_index < num_indices)
		{
			processBatch(shared);

			return true;
		}

		return false;
	}
};


#endif  // INCLUDED_CURE_PER_WARP_PATCH_CACHED_GEOMETRY_STAGE
