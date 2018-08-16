


#ifndef INCLUDED_CURE_TILE_RASTERIZER_MASK
#define INCLUDED_CURE_TILE_RASTERIZER_MASK

#pragma once

#include <utils.h>

#include "config.h"

#include "instrumentation.cuh"

#include "framebuffer.cuh"
#include "viewport.cuh"
#include "bitmask.cuh"

#include "work_assignment.cuh"

#include "rasterization_stage.cuh"


template <unsigned int NUM_WARPS, class BinTileSpace, class CoverageShader>
class TileRasterizerMask
{
private:
	static constexpr int stamp_bit_mask_x = BinTileSpace::StampsPerTileX;
	static constexpr int stamp_bit_mask_y = BinTileSpace::StampsPerTileY;

	static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

	
public:
	struct SharedMemT {};
	//static constexpr size_t SHARED_MEMORY = 0;

	typedef ::StampBitMask<BinTileSpace> StampBitMask;

	__device__
	static int run(SharedMemT& shared_memory, StampBitMask& mbitmask, int triangleId, int2 binid, int2 tileid)
	{
		Instrumentation::BlockObserver<6, 2> observer;

		int4 tile_bounds = BinTileSpace::tileBounds(binid, tileid);
		uchar4 rastColor = BinTileSpace::myRasterizerColor();

		auto tileSpace = BinTileSpace::template transformTile<RasterToClipConverter>(binid, tileid);

		StampBitMask bitmask;

		bitmask.setOn();
		#pragma unroll
		for (int e = 0; e < 3; ++e)
		{
			math::float3 edge = triangle_buffer.loadEdge(triangleId, e);
			// TODO: could also do that in geometry stage...
			float invx;
			if (edge.x != 0.0f)
				invx = 1.0f / edge.x;
			else
				invx = 1.0f / 0.000001f;
			int unsetRight = edge.x < 0.0f ? 1 : 0;

			//TODO: based on unsetRight we can also implement fill convention!
			float fillConvetion = -(0.008f - unsetRight * 0.016f);
			unsigned int linemask = (1U << stamp_bit_mask_x) - 1U;
			unsigned int switchmask = unsetRight * linemask;
			tileSpace.traverseStampsRows([=, &tileSpace, &bitmask](float y, int i)
			{
				// compute edge intersection
				float x = (-y*edge.y - edge.z)*invx;

				int t = min(stamp_bit_mask_x, tileSpace.stampFromX(x, 1.0f + fillConvetion));
				unsigned int tmask = (linemask >> (stamp_bit_mask_x - t)) ^ switchmask;
				bitmask.unmarkRow(i, tmask);
			}, 0.0f);
		}

		auto tile_coords = BinTileSpace::template tileCoords<RasterToClipConverter>(binid, tileid);
		CoverageShader coverage_shader { { tile_bounds.x , tile_bounds.y, tile_bounds.z, tile_bounds.w }, tile_coords.x, tile_coords.y, tile_coords.z, tile_coords.w };
		mbitmask = coverage_shader(bitmask);
		return bitmask.count();
		
	}
};

#endif  // INCLUDED_CURE_TILE_RASTERIZER_MASK
