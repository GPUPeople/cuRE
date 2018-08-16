


#ifndef INCLUDED_CURE_TILE_RASTERIZER
#define INCLUDED_CURE_TILE_RASTERIZER

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


template<int NUM_WARPS, int RUNS, FRAMEBUFFER_SYNC_METHOD SyncType = FRAMEBUFFER_SYNC_METHOD::NO_SYNC>
struct FramebufferAccess;

template<int NUM_WARPS, int RUNS>
struct FramebufferAccess<NUM_WARPS, RUNS, FRAMEBUFFER_SYNC_METHOD::NO_SYNC>
{
	struct SharedMemT
	{
	
	};

	__device__
	FramebufferAccess(SharedMemT& shared, int2 tile, int wip, int numWarps)
	{

	}

	template<typename F>
	__device__ 
		static void access(SharedMemT& shared, int2 tile, bool threadneedsaccess, int wip, int run, int numWarps, F f)
	{ 
		Instrumentation::BlockObserver<13, 3> observer;
		if (threadneedsaccess) 
			f(); 
	}
};

template<int NUM_WARPS, int RUNS>
struct FramebufferAccess<NUM_WARPS, RUNS, FRAMEBUFFER_SYNC_METHOD::SYNC_ALL>
{
	//whenever there is a need for any syncing, sync all, otherwise dont sync at all :)
	struct SharedMemT
	{
		unsigned int activetiles[NUM_WARPS];
		volatile bool needsync;
	};

	__device__
	FramebufferAccess(SharedMemT& shared, int2 tile, int wip, int numWarps)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		unsigned int myTileId = tile.x | (tile.y << 16);
		shared.activetiles[wip] = myTileId;
		shared.needsync = false;
		syncthreads(1, numWarps*WARP_SIZE);

		// figure out if there are any conflicts warps works on the same tile
		if (laneid() < wip)
			if (shared.activetiles[laneid()] == myTileId)
				shared.needsync = true;
	}

	template<typename F>
	__device__
	void access(SharedMemT& shared, int2 tile, bool threadneedsaccess, int wip, int run, int numWarps, F f)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		syncthreads(1, numWarps*WARP_SIZE);
		if (shared.needsync)
		{
#pragma unroll
			for (int i = 0; i < NUM_WARPS; ++i)
			{
				if (i == wip)
					if (threadneedsaccess)
						f();
				syncthreads(1, numWarps*WARP_SIZE);
			}
		}
		else
			if (threadneedsaccess)
				f();
	}
};

template<int NUM_WARPS, int RUNS>
struct FramebufferAccess<NUM_WARPS, RUNS, FRAMEBUFFER_SYNC_METHOD::SYNC_FRAGMENTS>
{
	//when starting the tile, figure out between which tiles we need to sync and then just sync them as we go along
	static_assert(NUM_WARPS <= 15, "The block synced access to the framebuffer cannot handle more than 15 warps");
	struct SharedMemT
	{
		cub::WarpScan<int>::TempStorage scan_storage;
		unsigned int activetiles[NUM_WARPS];
		int barrier[NUM_WARPS];
		int count[NUM_WARPS];
		int offset[NUM_WARPS];
	};

	__device__
	FramebufferAccess(SharedMemT& shared, int2 tile, int wip, int numWarps)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		unsigned int myTileId = tile.x | (tile.y << 16);
		shared.activetiles[wip] = myTileId;
		syncthreads(1, numWarps*WARP_SIZE);

		// figure out which other warps works on the same tile
		int active = 0, offset, count;
		if (laneid() < numWarps)
			active = (shared.activetiles[laneid()] == myTileId);

		// get overall number and when it is my turn
		cub::WarpScan<int>(shared.scan_storage).ExclusiveSum(active, offset, count);
		offset = __shfl_sync(~0U, offset, wip);

		//compute barrier id (id of the first warp)
		int barrier = 1 + __ffs(__ballot_sync(~0U, active)); //note that the id will start at 2

		//in case we really have 15 warps, we have to make sure not to use the 17th barrier
		//this could only happen if the last one is alone, instead we just use the 15th another time
		//the 15th cannot be used by anyone else but the 14th warp, and if the 15th warp does not access
		//the same tile as the 14th, the 14th will only use an arrive on the 15th barrier, and so will 
		//the 15th warp and two arrives with 32 threads each are fine too.
		static_assert(NUM_WARPS < 16, "nope");
		if (NUM_WARPS == 15)
			barrier = min(barrier, 15);

		shared.barrier[wip] = barrier;
		shared.offset[wip] = offset;
		shared.count[wip] = count;
	}

	template<typename F>
	__device__
	void access(SharedMemT& shared, int2 tile, bool threadneedsaccess, int wip, int run, int numWarps, F f)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		syncthreads(1, numWarps*WARP_SIZE);
		int count = shared.count[wip];
		int barrier = shared.barrier[wip];
		int offset = shared.offset[wip];
		//sequential access for every warp that wants to access
		for (int i = 0; i < offset; ++i)
			syncthreads(barrier, (count - i) * WARP_SIZE);
		if (threadneedsaccess)
			f();
		arrive(barrier, (count - offset) * WARP_SIZE);
	}
};


template<int NUM_WARPS, int RUNS>
struct FramebufferAccess<NUM_WARPS, RUNS, FRAMEBUFFER_SYNC_METHOD::SUBTILE_SYNC>
{
	//figure out when to sync between warps (i.e. on a subtile level, depending on the tile size: 64 fragments -> 2 tests)
	// and incorporate the information if threads want to write anything
	static_assert(NUM_WARPS <= 15, "The block synced access to the framebuffer cannot handle more than 15 warps");
	struct SharedMemT
	{
		unsigned int activetiles[NUM_WARPS];
		cub::WarpScan<int>::TempStorage scan_storage;
	};

	__device__
	FramebufferAccess(SharedMemT& shared, int2 tile, int wip, int numWarps)
	{

	}

	template<typename F>
	__device__
	void access(SharedMemT& shared, int2 tile, bool threadneedsaccess, int wip, int run, int numWarps, F f)
	{ 
		Instrumentation::BlockObserver<13, 3> observer;
		unsigned int myTileId = (!__any_sync(~0U, threadneedsaccess)) ? 0xFFFFFFFFU : tile.x | (tile.y << 16);
		shared.activetiles[wip] = myTileId;
		syncthreads(1, numWarps*WARP_SIZE);

		// figure out which other warps works on the same tile
		int active = 0, offset, count;
		if (laneid() < numWarps)
			active = (shared.activetiles[laneid()] == myTileId);
		syncthreads(1, numWarps*WARP_SIZE);

		if (myTileId == 0xFFFFFFFFU)
			return;

		// get overall number and when it is my turn
		cub::WarpScan<int>(shared.scan_storage).ExclusiveSum(active, offset, count);
		offset = __shfl_sync(~0U, offset, wip);

		//compute barrier id (id of the first warp)
		int barrier = 1 + __ffs(__ballot_sync(~0U, active)); //note that the id will start at 2

		//in case we really have 15 warps, we have to make sure not to use the 17th barrier
		//this could only happen if the last one is alone, instead we just use the 15th another time
		//the 15th cannot be used by anyone else but the 14th warp, and if the 15th warp does not access
		//the same tile as the 14th, the 14th will only use an arrive on the 15th barrier, and so will 
		//the 15th warp and two arrives with 32 threads each are fine too.
		static_assert(NUM_WARPS < 16, "nope");
		if (NUM_WARPS == 15)
			barrier = min(barrier, 15); 

		//sequential access for every warp that wants to access
		for (int i = 0; i < count; ++i)
		{
			if (i == offset)
			{
				//it's this warp's turn
				if (threadneedsaccess)
					f();
				arrive(barrier, (count - i) * WARP_SIZE);
				break;
			}
			else
				syncthreads(barrier, (count - i) * WARP_SIZE);
		}
	}
};


template<int NUM_WARPS, int RUNS>
struct FramebufferAccess<NUM_WARPS, RUNS, FRAMEBUFFER_SYNC_METHOD::MASK_SYNC>
{
	//for every shared tile, we keep a bitmask of active threads and sync as long as we need to serialize the individual accesses
	struct SharedMemT
	{
		union
		{
			unsigned int activetiles[NUM_WARPS];
			unsigned int bitmasks[NUM_WARPS];
		};
		int mymask[NUM_WARPS];
	};

	__device__
	FramebufferAccess(SharedMemT& shared, int2 tile, int wip, int numWarps)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		unsigned int myTileId = tile.x | (tile.y << 16);
		shared.activetiles[wip] = myTileId;
		syncthreads(1, numWarps*WARP_SIZE);

		// figure out which other warps works on the same tile
		int active = 0;
		if (laneid() < numWarps)
			active = (shared.activetiles[laneid()] == myTileId);

		//compute id of used bitmask
		shared.mymask[wip] = __ffs(__ballot_sync(~0U, active)) - 1;

	}

	template<typename F>
	__device__
	void access(SharedMemT& shared, int2 tile, bool threadneedsaccess, int wip, int run, int numWarps, F f)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		unsigned int myBit = 1 << laneid();

		syncthreads(1, numWarps*WARP_SIZE);
		shared.bitmasks[wip] = 0;
		unsigned int warp_mask = __ballot_sync(~0U, threadneedsaccess);
		while (syncthreads_or(warp_mask, 1, numWarps*WARP_SIZE))
		{
			unsigned int free;
			if (laneid() == 0)
				free = atomicOr(shared.bitmasks + shared.mymask[wip], warp_mask);
			free = __shfl_sync(~0U, ~free, 0);

			//if ((warp_mask & free) == 0)
			if (warp_mask & myBit & free)
			{ 
				//if (threadneedsaccess)
					f();
				//warp_mask = 0;
				threadneedsaccess = false;
			}

			warp_mask = __ballot_sync(~0U, threadneedsaccess);
			syncthreads(1, numWarps*WARP_SIZE);
			shared.bitmasks[wip] = 0;
		}
	}
};


template<int NUM_WARPS, int RUNS>
struct FramebufferAccess<NUM_WARPS, RUNS, FRAMEBUFFER_SYNC_METHOD::POLLED_MASK_SYNC>
{
	//for every shared tile, we keep a bitmask of active threads and lock via polling on a per thread basis
	struct SharedMemT
	{
		union
		{
			unsigned int activetiles[NUM_WARPS];
			unsigned int bitmasks[NUM_WARPS*RUNS];
		};
		int mymask[NUM_WARPS];
	};

	__device__
	FramebufferAccess(SharedMemT& shared, int2 tile, int wip, int numWarps)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		unsigned int myTileId = tile.x | (tile.y << 16);
		shared.activetiles[wip] = myTileId;
		syncthreads(1, numWarps*WARP_SIZE);

		// figure out which other warps works on the same tile
		int active = 0;
		if (laneid() < numWarps)
			active = (shared.activetiles[laneid()] == myTileId);

		//compute id of used bitmask
		shared.mymask[wip] = RUNS*(__ffs(__ballot_sync(~0U, active)) - 1);
		for (int i = 0; i < RUNS; ++i)
			shared.bitmasks[NUM_WARPS*wip + i] = 0;
		syncthreads(1, numWarps*WARP_SIZE);
	}

	template<typename F>
	__device__
	void access(SharedMemT& shared, int2 tile, bool threadneedsaccess, int wip, int run, int numWarps, F f)
	{
		Instrumentation::BlockObserver<13, 3> observer;
		unsigned int myBit = 1 << laneid();
		unsigned int warp_mask = __ballot_sync(~0U, threadneedsaccess);
		while (warp_mask)
		{
			unsigned int free;
			if (laneid() == 0)
				free = atomicOr(shared.bitmasks + shared.mymask[wip] + run, warp_mask);
			free = __shfl_sync(~0U, ~free, 0);

			if (warp_mask & myBit & free)
			{
				f();
				threadneedsaccess = false;
			}
			__threadfence_block();
			if (laneid() == 0)
				atomicAnd(shared.bitmasks + shared.mymask[wip] + run, free & warp_mask);
			warp_mask = __ballot_sync(~0U, threadneedsaccess);
		}
	}
};

template <unsigned int NUM_WARPS, bool SYNC_ACCESS, class BinTileSpace, class CoverageShader, class FragmentShader, class FrameBuffer, class BlendOp>
class TileRasterizer
{
private:
	using FragmentShaderInputs = typename ::FragmentShaderInfo<FragmentShader>::Inputs;

public:
	typedef ::TileBitMask<BinTileSpace> TileBitMask;
	typedef FramebufferAccess<NUM_WARPS, (BinTileSpace::StampsPerTileX*BinTileSpace::StampsPerTileY + WARP_SIZE - 1) / WARP_SIZE, SYNC_ACCESS ? TILE_RASTER_EXCLUSIVE_ACCESS_METHOD : FRAMEBUFFER_SYNC_METHOD::NO_SYNC> FBAccess;

	struct Triangle
	{
		struct
		{
			__align__(16) math::float3x3 M;
			__align__(16) math::float3 uz;
			FragmentShaderInputStorage<FragmentShaderInputs> fs_input;
		};
	};


	struct SharedMemT 
	{
		Triangle triangles[NUM_WARPS];
		FBAccess::SharedMemT fbaccess_shared;
	};

	__device__
	static void run(SharedMemT& shared_memory, int tilebit, int triangleId, int binX, int binY, int numTiles)
	{
		int2 tile_coords, global_tile;
		int wip = threadIdx.x / WARP_SIZE;
		Triangle& mytriangle = shared_memory.triangles[wip];

		{
			Instrumentation::BlockObserver<6, 2> observer;
		
			// find tile my tile
			//int2 tile_coords = binbitMask.getBitCoordsWarp(tile_offset);
			tile_coords = TileBitMask::bitToCoord(tilebit);

			global_tile = BinTileSpace::tileCoords(make_int2(binX, binY), tile_coords);
		
			//if (laneid() == 0 && (tile_coords_comp.x != tile_coords.x || tile_coords_comp.y != tile_coords.y))
			//	printf("not the same: %d %d vs %d %d\n", tile_coords.x, tile_coords.y, tile_coords_comp.x, tile_coords_comp.y);

			// load triangle to shared
			triangle_buffer.loadTriangleWarp(triangleId, &mytriangle.M, &mytriangle.uz);
			mytriangle.fs_input.loadWarp(triangle_buffer, triangleId);
			__threadfence_block();
		}

		FBAccess fba(shared_memory.fbaccess_shared, global_tile, wip, numTiles);

		BinTileSpace::traverseStampsWarp(make_int2(binX, binY), tile_coords, [&](int stamp, int tile_start_x, int tile_start_y, int x, int y, int part)
		{
			math::float3 p;
			float f1, f2, f3;
			{
				Instrumentation::BlockObserver<6, 2> observer;
				p = clipcoordsFromRaster(x, y);
				f1 = dot(mytriangle.M.row1(), p);
				f2 = dot(mytriangle.M.row2(), p);
				f3 = dot(mytriangle.M.row3(), p);
			}

			float z = -1.0f;
			math::float4 color;
			bool write = false;

			{
				Instrumentation::BlockObserver<13, 2> observer;

				if (DRAW_BOUNDING_BOX)
				{
					if (f1 >= 0.0f && f2 >= 0.0f && f3 >= 0.0f)
					{
						z = abs(dot(mytriangle.uz, p));// *0.5f + 0.5f;

						color = math::float4(z, z, z , 1.0f);
						write = z >= -1.0f && z < 1.0f;
					}
					else
					{
						color = math::float4(0, 0, 0, 1.0f);
						write = true;
					}
				}
				else
				{
					if (f1 >= 0.0f && f2 >= 0.0f && f3 >= 0.0f)
					{
						z = dot(mytriangle.uz, p);

						if (z >= -1.0f && z <= 1.0f)  // clipping!
						{
							{
								math::float3 uw = math::float3(1.0f, 1.0f, 1.0f) * mytriangle.M;

								float rcpw = dot(uw, p);
								float w = 1.0f / rcpw;

								math::float3 u = math::float3(f1 * w, f2 * w, f3 * w);

								FragmentShader shader { { x, y }, { p.x, p.y, z, rcpw }, u, { f1 / length(mytriangle.M.row1().xy()), f2 / length(mytriangle.M.row2().xy()), f3 / length(mytriangle.M.row3().xy()) }, triangleId };
								color = callFragmentShader(shader, mytriangle.fs_input, u);
								write = !shader.discarded();
							}
						}
					}
				}
			}

			fba.access(shared_memory.fbaccess_shared, make_int2(tile_start_x, tile_start_y), write, wip, part, numTiles, [write, x, y, z, color]()
			{
				if (write)
				{
					float z_dest = FrameBuffer::readDepth(x, y);

					if (!DEPTH_TEST || z < z_dest)
					{
						if (DEPTH_WRITE)
							FrameBuffer::writeDepth(x, y, z);
						FrameBuffer::template writeColor<BlendOp>(x, y, color);
					}
				}
			});
		});
	}
};

#endif  // INCLUDED_CURE_TILE_RASTERIZER
