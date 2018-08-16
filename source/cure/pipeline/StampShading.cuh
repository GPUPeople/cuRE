
#ifndef INCLUDED_CURE_STAMP_SHADING
#define INCLUDED_CURE_STAMP_SHADING

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "config.h"

#include "instrumentation.cuh"

#include "fragment_shader_stage.cuh"

#include "viewport.cuh"
#include "bitmask.cuh"

#include "rasterization_stage.cuh"

#include <ptx_primitives.cuh>
#include <cub/cub.cuh>

template<unsigned int NUM_WARPS, class BinTileSpace, FRAMEBUFFER_SYNC_METHOD METHOD>
struct StampShadingExclusiveAccess;


template<unsigned int NUM_WARPS, class BinTileSpace>
struct StampShadingExclusiveAccess<NUM_WARPS, BinTileSpace, FRAMEBUFFER_SYNC_METHOD::NO_SYNC>
{
	struct SharedMemT
	{
	};
	
	template<typename F>
	__device__
	StampShadingExclusiveAccess(SharedMemT& shared_memory, StampBitMask<BinTileSpace>* stamp_bit_masks, int stampmaskoffset, int bitid, int bit, int localoffset, int startTile, int numStamps, F tileInfo)
	{
	}

	template<typename F>
	__device__
	void access(SharedMemT& shared_memory, F blend)
	{
		blend();
	}
};

template<unsigned int NUM_WARPS, class BinTileSpace, bool EXACT_WARP_COUNT>
struct StampShadingExclusiveAccessCountDistributor
{
	static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
	static constexpr int SEARCH_GROUP_SIZE = WARP_SIZE / 4;
	static constexpr int NUM_SEARCH_GROUPS = NUM_THREADS / SEARCH_GROUP_SIZE;

	struct SharedMemT
	{
		int maxTile;
		volatile int done_tiles[NUM_THREADS + 1];
	};

	int myoverlap;
	int leader;
	int mystampmaskoffset;

	template<typename F>
	__device__
	StampShadingExclusiveAccessCountDistributor(SharedMemT& shared_memory, StampBitMask<BinTileSpace>* stamp_bit_masks, int stampmaskoffset, int bitid, int bit, int localoffset, int startTile, int numStamps, F tileInfo)
	{
		Instrumentation::BlockObserver<13, 2> observer;
		int lid = laneid();
		int wp = threadIdx.x / WARP_SIZE;
		if (threadIdx.x == numStamps - 1)
		{
			shared_memory.maxTile = stampmaskoffset + 1;
			shared_memory.done_tiles[NUM_THREADS] = 0;
		}

		__syncthreads();

		int group = threadIdx.x / SEARCH_GROUP_SIZE;
		int gid = GroupId<SEARCH_GROUP_SIZE>();

		//use warps to search to the front
		for (int i = startTile + group; i < shared_memory.maxTile; i += NUM_SEARCH_GROUPS)
		{

			int found = 0;
			auto thisTileInfo = tileInfo(i);
			StampBitMask<BinTileSpace> thisBitMask = stamp_bit_masks[i];
			//quickly skip those which have an empty mask..
			if (thisBitMask.count() == 0)
				continue;
			for (int j = i - gid - 1; j >= startTile; j -= static_cast<int>(SEARCH_GROUP_SIZE))
			{
				//check for overlap
				if (tileInfo(j) == thisTileInfo && thisBitMask.overlap(stamp_bit_masks[j]))
					found = j + 1;
				if (anyGroup<SEARCH_GROUP_SIZE>(found))
					break;
			}
			//write info to shared
			unsigned int mask = ballotGroup<SEARCH_GROUP_SIZE>(found != 0);
			int first = __ffs(mask);
			if (first > 0)
			{
				int closestsOverlap = __shfl_sync(~0U, found, first - 1, SEARCH_GROUP_SIZE) - 1;
				shared_memory.done_tiles[i] = closestsOverlap;
			}
			else
				shared_memory.done_tiles[i] = NUM_THREADS;
		}
		__syncthreads();

		mystampmaskoffset = stampmaskoffset;
		myoverlap = shared_memory.done_tiles[stampmaskoffset];

		//we need the sync here as thread for one stamp might be scatter over multiple warps...
		__syncthreads();

		if (threadIdx.x < numStamps)
		{
			//find leader
			int stamps = stamp_bit_masks[stampmaskoffset].count();
			leader = max(lid - localoffset, 0);

			//figure out over how many warps the tile spans and compute how many warps need to touch it before it is done
			int inelement = stamps - bitid - 1;
			int beforeWarp = inelement - lid + WARP_SIZE - 1;
			int afterWarp = stamps + lid - inelement - 1;
			int sumwarps = 1 + min(wp, beforeWarp / WARP_SIZE) + afterWarp / WARP_SIZE;
			//note that that computation is potentially wrong for the last tile as threads will only work during the next round
			//however, nobody will wait for the last anyway

			//use as done counter now
			shared_memory.done_tiles[stampmaskoffset] = EXACT_WARP_COUNT ? sumwarps : 1;
		}

		__syncthreads();
	}
};

template<unsigned int NUM_WARPS, class BinTileSpace>
struct StampShadingExclusiveAccess<NUM_WARPS, BinTileSpace, FRAMEBUFFER_SYNC_METHOD::SYNC_ALL> : public StampShadingExclusiveAccessCountDistributor<NUM_WARPS, BinTileSpace, true>
{

	template<typename F>
	__device__
	StampShadingExclusiveAccess(SharedMemT& shared_memory, StampBitMask<BinTileSpace>* stamp_bit_masks, int stampmaskoffset, int bitid, int bit, int localoffset, int startTile, int numStamps, F tileInfo)
		: StampShadingExclusiveAccessCountDistributor(shared_memory, stamp_bit_masks, stampmaskoffset, bitid, bit, localoffset, startTile, numStamps, tileInfo)
	{
	}

	template<typename F>
	__device__
	void access(SharedMemT& shared_memory, F blend)
	{
		//blend();

		//we need to fool the compiler into thinking that blending could happen more than once,
		//so that it does not wait for all threads to be allowed to blend
		bool hasblend = false;
		while (__any_sync(~0U, !hasblend))
		{
			bool nowblend = !hasblend && shared_memory.done_tiles[myoverlap] == 0;
			if (__shfl_sync(~0U, nowblend, leader))
			{
				blend();
				__threadfence_block();
				if (laneid() == leader)
				{
					//int r = 
						atomicSub(const_cast<int*>(shared_memory.done_tiles) + mystampmaskoffset, 1);
					//if (r <= 0 || r > 3)
					//	printf("that should not happen: %d %d\n", mystampmaskoffset, r);
					__threadfence_block();
				}
				hasblend = true;
			}
			else
				__threadfence();
		}
	}

};


template<unsigned int NUM_WARPS, class BinTileSpace>
struct StampShadingExclusiveAccess<NUM_WARPS, BinTileSpace, FRAMEBUFFER_SYNC_METHOD::SYNC_FRAGMENTS> : public StampShadingExclusiveAccessCountDistributor<NUM_WARPS, BinTileSpace, false>
{
	int ActiveWarps;

	template<typename F>
	__device__
	StampShadingExclusiveAccess(SharedMemT& shared_memory, StampBitMask<BinTileSpace>* stamp_bit_masks, int stampmaskoffset, int bitid, int bit, int localoffset, int startTile, int numStamps, F tileInfo)
		: StampShadingExclusiveAccessCountDistributor(shared_memory, stamp_bit_masks, stampmaskoffset, bitid, bit, localoffset, startTile, numStamps, tileInfo)
	{
		ActiveWarps = (numStamps + WARP_SIZE - 1) / WARP_SIZE;
	}

	template<typename F>
	__device__
	void access(SharedMemT& shared_memory, F blend)
	{
		//blend();

		//we need to fool the compiler into thinking that blending could happen more than once,
		//so that it does not wait for all threads to be allowed to blend
		bool hasblend = false;
		do
		{
			bool nowblend = !hasblend && shared_memory.done_tiles[myoverlap] == 0;
			if (nowblend)
			{
				blend();
				__threadfence_block();
				if (laneid() == leader)
					shared_memory.done_tiles[mystampmaskoffset] = 0;
				hasblend = true;
			}
		} while (syncthreads_or(!hasblend, 2, ActiveWarps * 32) != 0);
	}
};




template <unsigned int NUM_WARPS, FRAMEBUFFER_SYNC_METHOD EXCLUSIVE_ACCESS, bool QUAD_SHADING, class BinTileSpace, class CoverageShader, class FragmentShader, class FrameBuffer, class BlendOp>
class StampShading
{
	using FragmentShaderInputs = typename ::FragmentShaderInfo<FragmentShader>::Inputs;
	
public:

	typedef StampShadingExclusiveAccess<NUM_WARPS, BinTileSpace, EXCLUSIVE_ACCESS> AccessControl;

	struct SharedMemT 
	{
		AccessControl::SharedMemT access_control;
	};

	template<typename F>
	__device__
	static void run(SharedMemT& shared_memory, int triangleId, int2 fragment, StampBitMask<BinTileSpace>* stamp_bit_masks, int stampmaskoffset, int bitid, int bit, int localoffset, int startTile, int numStamps, F tileInfo)
	{
		AccessControl ac(shared_memory.access_control, stamp_bit_masks, stampmaskoffset, bitid, bit, localoffset, startTile, numStamps, tileInfo);
		

		if (threadIdx.x < numStamps)
		{ 
			math::float3x3 M;
			math::float3 uz;
			math::int4 bounds;
			math::float4 color;
			float z;
			bool write = false;

			{
				Instrumentation::BlockObserver<16, 2> observer;
			
				triangle_buffer.loadTriangle(triangleId, &M, &uz, &bounds);

				math::float3 p = clipcoordsFromRaster(fragment.x, fragment.y);

				z = dot(uz, p);

				//if (z >= -1.0f && z <= 1.0f)  // clipping!
				if (z >= -1.0f )  // clipping!
				{
					//TODO: check if that loading is right...
					FragmentShaderInputStorage<FragmentShaderInputs> fs_input;
					fs_input.load(triangle_buffer, triangleId);

					float f1 = dot(M.row1(), p);
					float f2 = dot(M.row2(), p);
					float f3 = dot(M.row3(), p);


					math::float3 uw = math::float3(1.0f, 1.0f, 1.0f) * M;
					float rcpw = dot(uw, p);
					float w = 1.0f / rcpw;

					math::float3 u = math::float3(f1 * w, f2 * w, f3 * w);

					FragmentShader shader{ { fragment.x, fragment.y }, { p.x, p.y, z, rcpw }, u, { f1 / length(M.row1().xy()), f2 / length(M.row2().xy()), f3 / length(M.row3().xy()) }, triangleId };
					color = callFragmentShader(shader, fs_input, u);
					write = !shader.discarded();
				}
			}
			{
				Instrumentation::BlockObserver<13, 2> observer;
				ac.access(shared_memory.access_control, [&]()
				{
					if (write)
					{
						if (!QUAD_SHADING || (((stamp_bit_masks[stampmaskoffset].mask >> bit) & 0x1) == 0x1))
						{
							float z_dest = FrameBuffer::readDepth(fragment.x, fragment.y);

							if (!DEPTH_TEST || z < z_dest)
							{
								if (DEPTH_WRITE)
									FrameBuffer::writeDepth(fragment.x, fragment.y, z);
								FrameBuffer::template writeColor<BlendOp>(fragment.x, fragment.y, color);
							}
						}
					}
				});
			}
		}
	}
};

#endif  // INCLUDED_CURE_TILE_RASTERIZER
