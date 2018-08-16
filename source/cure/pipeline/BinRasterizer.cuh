


#ifndef INCLUDED_CURE_BIN_RASTERIZER
#define INCLUDED_CURE_BIN_RASTERIZER

#pragma once

#include <utils.h>

#include "config.h"

#include "instrumentation.cuh"

#include "framebuffer.cuh"
#include "viewport.cuh"
#include "bitmask.cuh"

#include "work_assignment.cuh"

#include "rasterization_stage.cuh"


template <unsigned int NUM_WARPS, class BinTileSpace>
class BinRasterizer
{
private:
	static constexpr int tile_bit_mask_x = BinTileSpace::TilesPerBinX;
	static constexpr int tile_bit_mask_y = BinTileSpace::TilesPerBinY;

	static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

	
public:
	struct SharedMemT {};
	//static constexpr size_t SHARED_MEMORY = 0;

	typedef ::TileBitMask<BinTileSpace> TileBitMask;

	__device__
	static int run(SharedMemT& shared_memory, TileBitMask& mbitmask, int triangleId, int2 binid, const math::uint4& tribounds)
	{
		Instrumentation::BlockObserver<5, 2> observer;

		int4 binBounds = BinTileSpace::binBounds(binid);
		uchar4 rastColor = BinTileSpace::myRasterizerColor();

		auto binSpace = BinTileSpace::template transformBin<RasterToClipConverter>(binid);

		TileBitMask bitmask;

#if 0
		bitmask.setOn();
#else
		// mask bounding box
		int lowerX = min(tile_bit_mask_x, max(0, BinTileSpace::relativeTileHitX(binid, tribounds.x)));
		int upperX = min(tile_bit_mask_x, max(0, BinTileSpace::relativeTileHitX(binid, tribounds.z)+1));
		bitmask.repeatRow(lowerX, upperX);
		int lowerY = min(tile_bit_mask_y, max(0, BinTileSpace::relativeTileHitY(binid, tribounds.y)));
		int upperY = min(tile_bit_mask_y, max(0, BinTileSpace::relativeTileHitY(binid, tribounds.w) + 1));
		bitmask.andStride(lowerY*tile_bit_mask_x, upperY*tile_bit_mask_x);
#endif

		if (!DRAW_BOUNDING_BOX)
		{
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
				float off1 = edge.y > 0.0f;
				int unsetRight = edge.x < 0.0f;

				unsigned int linemask = (1U << tile_bit_mask_x) - 1U;
				unsigned int switchmask = unsetRight * linemask;
				binSpace.traverseTileRows([binBounds, &binSpace, &bitmask, linemask, edge, invx, switchmask, unsetRight, rastColor](float y, int i)
				{
					// compute edge intersection
					float x = (-y*edge.y - edge.z)*invx;

					int t = min(tile_bit_mask_x, binSpace.tileFromX(x) + unsetRight);
					unsigned int tmask = (linemask >> (tile_bit_mask_x - t)) ^ switchmask;
					bitmask.unmarkRow(i, tmask);

					//// just draw that for now
					//math::float2 p = rastercoordsFromClip(x, y);
					//FrameBuffer::writeColor(min(binBounds.z, max(binBounds.x, static_cast<int>(p.x + 0.5f))), p.y + 0.5f, make_uchar4(0,0,0,255));
				}, false, off1);
			}
		}

		//math::int4 b = triangle_buffer.loadBounds(triangleId);
		//FrameBuffer::writeColor((b.x + b.z) / 2, (b.y + b.w) / 2, rastColor);

		//return 0;

		mbitmask = bitmask;
		return bitmask.count();
		

		//// color the tiles for now
		//BinTileSpace::traverseTiles(binid, [rastColor, bitmask](int tile, int2 tileid, int4 tilebounds){
		//	if (bitmask.isset(tileid.x, tileid.y))
		//		for (int x = tilebounds.x; x < tilebounds.z; ++x)
		//			for (int y = tilebounds.y; y < tilebounds.w; ++y)
		//				FrameBuffer::writeColor(x, y, rastColor);
		//});


		//return 0;

		////int x = (binBounds.x + binBounds.z) / 2;
		////int y = (binBounds.y + binBounds.w) / 2;
		//for (int y = binBounds.y; y < binBounds.w; ++y)
		//{
		//	for (int x = binBounds.x; x < binBounds.z; ++x)
		//	{
		//		FrameBuffer::writeColor(x, y, rastColor);
		//	}
		//}
	}
};

#endif  // INCLUDED_CURE_BIN_RASTERIZER
