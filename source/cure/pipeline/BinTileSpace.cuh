
#ifndef INCLUDED_CURE_BIN_TILE_SPACE
#define INCLUDED_CURE_BIN_TILE_SPACE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include <meta_utils.h>
#include <utils.h>
#include <cstdint>
#include "config.h"

struct BlockRasterizerId
{
	__device__
	static int rasterizer()
	{
		return blockIdx.x;
	}
};

template <int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId = BlockRasterizerId>
class TileSpace
{
protected:
	static constexpr int bin_width = bin_size_x * stamp_width;
	static constexpr int bin_height = bin_size_y * stamp_height;

	static constexpr int num_bins_x = x_max / bin_width;
	static constexpr int num_bins_y = y_max / bin_height;

	//static_assert(bin_size_x*bin_size_y%WARP_SIZE == 0, "ERROR: bin size must be a multiple of the warpsize");
	static_assert(stamp_width*stamp_height%WARP_SIZE == 0, "ERROR: stamp size must be a multiple of the warpsize");

	__device__
	static int rasterizer()
	{
		return RasterizerId::rasterizer();
	}

public:

	static constexpr int TilesPerBinX = bin_size_x;
	static constexpr int TilesPerBinY = bin_size_y;
	static constexpr int StampsPerTileX = stamp_width;
	static constexpr int StampsPerTileY = stamp_height;

	//typedef CoverageMask<bin_size_x, bin_size_y> BinCoverageMask;
	//typedef CoverageMask<stamp_width, stamp_height> StampCoverageMask;


	__device__
	static int MyQueue()
	{
		return rasterizer();
	}


	__device__
	static int num_rasterizers()
	{
		return NUM_BLOCKS;
	}

	__device__
	static int2 bin(int x, int y)
	{
		return make_int2(x / bin_width, y / bin_height);
	}

	__device__
	static int left(int i)
	{
		return i * bin_width;
	}

	__device__
	static int top(int j)
	{
		return j * bin_height;
	}

	__device__
	static int right(int i)
	{
		return left(i) + bin_width;
	}

	__device__
	static int bottom(int j)
	{
		return top(j) + bin_height;
	}

	__device__
	static int rasterizer(int x, int y)
	{
		return (y + x) % NUM_BLOCKS;
	}

	__device__
	static int rasterizer(int2 b)
	{
		return (b.y + b.x) % NUM_BLOCKS;
	}

	__device__
	static int4 binBounds(int2 bin)
	{
		return make_int4(bin.x * bin_width, bin.y * bin_height, (bin.x + 1) * bin_width, (bin.y + 1)* bin_height);
	}

	__device__
	static int4 tileBounds(int2 bin, int2 tile)
	{
		int x = left(bin.x) + tile.x * stamp_width;
		int y = top(bin.y) + tile.y * stamp_height;
		return make_int4(x, y, x + stamp_width, y + stamp_height);
	}

	__device__
	static int4 tileBounds(int2 bin, int tile)
	{
		int x = left(bin.x) + (tile % bin_size_x) * stamp_width;
		int y = top(bin.y) + (tile / bin_size_x) * stamp_height;
		return make_int4(x, y, x + stamp_width, y + stamp_height);
	}

	__device__
	static int2 tileCoords(int2 bin, int2 tile)
	{
		return make_int2(bin.x * bin_size_x + tile.x,
			bin.y * bin_size_y + tile.y);
	}


	__device__
	static int localTileHitX(int2 bin, float x)
	{
		int l = left(bin.x);
		float diff = (x - l) / static_cast<float>(stamp_width);
		int h = 1.0f + max(-1.0f, min(static_cast<float>(stamp_width), diff));
		return h - 1;
	}
	__device__
	static int relativeTileHitX(int2 bin, int x)
	{
		int l = left(bin.x);
		int diff = (x - l) / stamp_width;
		return diff;
	}

	__device__
	int  localTileHitY(int2 bin, float y)
	{
		int t = top(bin.y);
		float diff = (y - t) / static_cast<float>(stamp_height);
		int h = 1.0f + max(-1.0f, min(static_cast<float>(stamp_height), diff));
		return h - 1;
	}
	__device__
	static int relativeTileHitY(int2 bin, int y)
	{
		int t = top(bin.y);
		int diff = (y - t) / stamp_height;
		return diff;
	}


	__device__
	static int relativeStampHitX(int2 bin, int2 tile, int x)
	{
		int l = left(bin.x) + tile.x * stamp_width;
		int diff = (x - l);
		return diff;
	}
	__device__
	static int relativeStampHitY(int2 bin, int2 tile, int y)
	{
		int t = top(bin.y) + tile.y * stamp_height;
		int diff = (y - t);
		return diff;
	}


	template <typename F>
	__device__ 
	static void  traverseTileRows(int2 bin, F f, bool flipY = false, int startoffset = 0)
	{
		int i_flip = flipY;
		int flipperY = 1 - 2 * static_cast<int>(flipY);
		int t = top(bin.y) + startoffset * stamp_height;
		int y = t + i_flip * bin_height;
		#pragma unroll
		for (int i = 0; i < bin_size_y; ++i, y += flipperY * stamp_height)
			f(y, i);
	}


	template<class CoordTransform>
	class TransformedBin
	{
		math::float2 start;
		math::float2 stamp_size;
		math::float2 inv_stamp_size;
	public:
		__device__
		TransformedBin(int left, int top) :
		start(CoordTransform::point(math::float2(left-0.5f, top-0.5f))),
			stamp_size(CoordTransform::vec(math::float2(stamp_width, stamp_height))),
			inv_stamp_size(1.0f / stamp_size.x, 1.0f / stamp_size.y)
		{}
	
		template <typename F>
		__device__
		void  traverseTileRows(F f, bool flipY = false, float startoffset = 0.0f)
		{
			float f_flip = flipY;
			float flipperY = 1.0f - 2.0f * static_cast<float>(flipY);
			float t = start.y + startoffset * stamp_size.y;
			float y = t + f_flip * stamp_size.y * bin_size_y;
			float step = flipperY * stamp_size.y;
			#pragma unroll
			for (int i = 0; i < bin_size_y; ++i, y += step)
				f(y, i);
		}

		__device__
		int tileFromX(float x)
		{
			float diff = (x - start.x)* inv_stamp_size.x;
			int h = 1.0f + max(-1.0f, min(static_cast<float>(bin_size_x), diff));
			return h - 1;
		}

		__device__
		int tileFromY(float y)
		{
			float diff = (y - start.y)* inv_stamp_size.y;
			int h = 1.0f + max(-1.0f, min(static_cast<float>(bin_size_y), diff));
			return h - 1;
		}
	};
	
	template<class CoordTransform>
	__device__
	static TransformedBin<CoordTransform> transformBin(int2 bin)
	{
		return TransformedBin<CoordTransform>(left(bin.x), top(bin.y));
	}


	template<class CoordTransform>
	class TransformedTile
	{
		math::float2 start;
		math::float2 fragment_size;
		math::float2 inv_fragment_size;
	public:
		__device__
		TransformedTile(int left, int top) :
			start(CoordTransform::point(math::float2(left, top))),
			fragment_size(CoordTransform::vec(math::float2(1.0f, 1.0f))),
			inv_fragment_size(1.0f / fragment_size.x, 1.0f / fragment_size.y)
		{}

		template <typename F>
		__device__
		void traverseStampsRows(F f, float startoffset = 0.0f)
		{
			float y = start.y + startoffset * fragment_size.y;
			float step = fragment_size.y;
#pragma unroll
			for (int i = 0; i < stamp_height; ++i, y += step)
				f(y, i);
		}

		__device__
		int stampFromX(float x, float addoffset = 0)
		{
			float diff = (x - start.x)* inv_fragment_size.x + addoffset;
			int h = 1.0f + max(-1.0f, min(static_cast<float>(stamp_width), diff));
			return h - 1;
		}

		__device__
		int stampFromY(float y, float addoffset = 0)
		{
			float diff = (y - start.y)* inv_fragment_size.y + addoffset;
			int h = 1.0f + max(-1.0f, min(static_cast<float>(stamp_height), diff));
			return h - 1;
		}
	};

	template<class CoordTransform>
	__device__
	static TransformedTile<CoordTransform> transformTile(int2 bin, int2 tile)
	{
		return TransformedTile<CoordTransform>(left(bin.x) + tile.x * stamp_width, top(bin.y) + tile.y * stamp_height);
	}

	template<class CoordTransform>
	__device__
	static math::float4 tileCoords(int2 bin, int2 tile)
	{
		auto lt = CoordTransform::point({ static_cast<float>(left(bin.x) + tile.x * stamp_width), static_cast<float>(top(bin.y) + tile.y * stamp_height) });
		auto rb = CoordTransform::point({ static_cast<float>(left(bin.x) + tile.x * stamp_width + stamp_width), static_cast<float>(top(bin.y) + tile.y * stamp_height + stamp_height) });
		return { lt.x, lt.y, rb.x, rb.y };
	}

	template <typename F>
	__device__
	static void traverseTiles(int2 bin, F f)
	{
		int2 p = make_int2(left(bin.x), top(bin.y));

		for (int j = 0; j < bin_size_y; ++j)
			for (int i = 0; i < bin_size_x; ++i)
			{
				int x = p.x + i * stamp_width;
				int y = p.y + j * stamp_height;

				f(bin_size_x*j + i, make_int2(i, j), make_int4(x, y, x + stamp_width, y + stamp_height));
			}
	}
	
	template <typename F>
	__device__
	static void traverseTilesWarp(int2 bin, F f)
	{
		int2 p = make_int2(left(bin.x), top(bin.y));
		#pragma unroll
		for (int toffset = 0; toffset < bin_size_x * bin_size_y; toffset += WARP_SIZE)
		{
			int tile = toffset + laneid();
			int j = tile / bin_size_x;
			int i = tile % bin_size_x;
			int x = p.x + i * stamp_width;
			int y = p.y + j * stamp_height;

			f(tile, make_int4(x, y, x + stamp_width, y + stamp_height));
		}
	}

	template <typename F>
	__device__
	static void traverseStampWarp(int2 tile, F f)
	{
		#pragma unroll
		for (int sample = 0; sample < stamp_width * stamp_height; sample += WARP_SIZE)
		{
			int s = sample + laneid();
			int j = s / stamp_width;
			int i = s % stamp_width;
			int x = tile.x + i;
			int y = tile.y + j;

			f(x, y);
		}
	}

	template <typename F>
	__device__
	static void traverseStampsWarp(int2 bin, int2 tile, F f)
	{
		int px = left(bin.x) + tile.x * stamp_width;
		int py = top(bin.y) + tile.y * stamp_height;

		int part = 0;
		#pragma unroll
		for (int toffset = 0; toffset < stamp_width * stamp_height; toffset += WARP_SIZE)
		{
			int stamp = toffset + laneid();
			int j = stamp / stamp_width;
			int i = stamp % stamp_width;
			int x = px + i;
			int y = py + j;

			f(stamp, px, py, x, y, part);
			++part;
		}
	}

	__device__
	static uchar4 myRasterizerColor()
	{
		return make_uchar4(rasterizer() % 5 * 50, rasterizer() / 5 % 5 * 50, rasterizer() / 25 % 5 * 50, 255);
	}
};


template <PATTERNTECHNIQUE technique, int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId = BlockRasterizerId>
class PatternTileSpace {};

template < int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId>
class PatternTileSpace <PATTERNTECHNIQUE::DIAGONAL, NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
	: public TileSpace<NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
{
public:
	__device__
	static int numHitBinsForMyRasterizer(const int2 start, const int2 end)
	{
			const int r = rasterizer();
			const int w = end.x - start.x + 1;
			const int h = end.y - start.y + 1;

			int full_elements_x = w / NUM_BLOCKS;
			int full_elements_y = h / NUM_BLOCKS;
			int right = w - full_elements_x * NUM_BLOCKS;
			int top = h - full_elements_y * NUM_BLOCKS;
			int full_sums = full_elements_x * h + full_elements_y * right;

			int start_offset = (NUM_BLOCKS + r + 1 - rasterizer(start)) % NUM_BLOCKS;
			int first_hit_diag = min(right, start_offset) - min(right, max(start_offset - top, 0));
			int second_hit_diag = right - min(right, start_offset + NUM_BLOCKS - top);
			int res = full_sums + first_hit_diag + second_hit_diag;

			return res;
		}


	__device__
	static int2 getHitBinForMyRasterizer(int i, const int2 start, const int2 end)
	{
			const int r = rasterizer();
			const int w = end.x - start.x + 1;
			const int h = end.y - start.y + 1;

			int full_elements_x = w / NUM_BLOCKS;
			int full_elements_y = h / NUM_BLOCKS;
			int right = w - full_elements_x * NUM_BLOCKS;
			int top = h - full_elements_y * NUM_BLOCKS;

			int start_offset = (NUM_BLOCKS + r + 1 - rasterizer(start)) % NUM_BLOCKS;

			int x, y;
			int region0bound = full_elements_y * (full_elements_x * NUM_BLOCKS + right);
			int region1bound = region0bound + min(start_offset, top);
			if (i < region0bound)
			{
				x = i % (full_elements_x * NUM_BLOCKS + right);
				int row = i / (full_elements_x * NUM_BLOCKS + right);
				int yoffset = (NUM_BLOCKS - start_offset + x) % NUM_BLOCKS;
				y = (row + 1) * NUM_BLOCKS - yoffset - 1;
			}
			else if (i < region1bound)
			{
				x = i + start_offset - region1bound;
				y = full_elements_y * NUM_BLOCKS + start_offset - x - 1;
			}
			else
			{
				int k = i - region1bound;
				int x_block = k / top;
				int x_local = k - x_block * top;
				x = (x_block + 1) * NUM_BLOCKS + start_offset - top + x_local;
				y = full_elements_y * NUM_BLOCKS + top - 1 - x_local;
			}

			return make_int2(start.x + x, start.y + y);
		}


	template <typename F>
	__device__
	static unsigned int traverseRasterizers(int2 start, int2 end, F f)
	{
			int r = rasterizer(start);

			int num = end.x - start.x + end.y - start.y + 1;
			int endr = min(r + num, NUM_BLOCKS);
			for (int i = r; i < endr; ++i)
				f(i);
			for (int i = 0; i < min(r + num - endr, r); ++i)
				f(i);

			return min(num, NUM_BLOCKS);

			//unsigned int c = 0;

			//for (int b = start.y; b <= end.y && c < NUM_BLOCKS; ++b, ++c)
			//{
			//	f(r);
			//	r = (r + 1) >= NUM_BLOCKS ? 0 : r + 1;
			//}

			//for (int b = start.x + 1; b <= end.x && c < NUM_BLOCKS; ++b, ++c)
			//{
			//	f(r);
			//	r = (r + 1) >= NUM_BLOCKS ? 0 : r + 1;
			//}

			//return c;
		}
};

template < int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId>
class PatternTileSpace <PATTERNTECHNIQUE::OFFSET, NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
	: public TileSpace<NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
{
private:
	static constexpr int PARTIAL = OFFSET_PARAMETER;
	static constexpr int num_rasterizers_s = NUM_BLOCKS*PARTIAL;
	static constexpr int def_shift_per_line = (NUM_BLOCKS + PARTIAL - 1) / PARTIAL;

public:
	__device__
		//static int numHitBinsForMyRasterizer(const int2 start, const int2 end)
	static int numHitBinsForMyRasterizer(const int2 bb_from, const int2 end)
	{
		//maybe rename later
		const int2 bb_to = { end.x + 1, end.y + 1 };
		//

		const int rasterizer_id = rasterizer();

		const int w = bb_to.x - bb_from.x;
		const int h = bb_to.y - bb_from.y;

		const int full_sets_per_line = w / NUM_BLOCKS;
		const int found_full = full_sets_per_line * h;

		const int bb_from_s = (bb_from.x + full_sets_per_line * NUM_BLOCKS)*PARTIAL;
		const int bb_max_s = bb_to.x * PARTIAL - 1;
		const int rasterizer_id_s = rasterizer_id*PARTIAL;

		const int step = (bb_from_s - rasterizer_id_s + (NUM_BLOCKS + 1)*NUM_BLOCKS - 1) / NUM_BLOCKS - NUM_BLOCKS;

		int part_start_column_s = rasterizer_id_s + step*NUM_BLOCKS;
		int part_start_row = (-step + num_bins_y * PARTIAL + 1);
		int possible_per_rep = ((bb_max_s - part_start_column_s + NUM_BLOCKS) / NUM_BLOCKS);

		int lines_big = part_start_row - bb_from.y;
		int lines_small = part_start_row - bb_to.y;
		int seq_reps_big = lines_big / PARTIAL;
		int seq_reps_small = lines_small / PARTIAL;

		int found_rem_big = (seq_reps_big*possible_per_rep + min(possible_per_rep, lines_big - seq_reps_big*PARTIAL));
		int found_rem_small = (seq_reps_small*possible_per_rep + min(possible_per_rep, lines_small - seq_reps_small*PARTIAL));
		int found_remaining = found_rem_big - found_rem_small;

		return (found_full + found_remaining);
	}


	__device__
	static int2 getHitBinForMyRasterizer(int i, const int2 bb_from, const int2 end)
	{
		//maybe rename later
		const int2 bb_to = { end.x + 1, end.y + 1 };
		//

		const int rasterizer_id = rasterizer();

		int bb_from_s = bb_from.x * PARTIAL;
		int bb_max_s = bb_to.x * PARTIAL - 1;
		int rasterizer_id_s = rasterizer_id*PARTIAL;

		int w = bb_to.x - bb_from.x;
		int h = bb_to.y - bb_from.y;

		int full_sets_per_line = w / NUM_BLOCKS;
		int found_full = full_sets_per_line * h;

		int X = 0;
		int Y = 0;

		if (i < found_full)
		{
			int y_times = i / full_sets_per_line;
			Y = bb_to.y - y_times - 1;
			int step_s = (rasterizer_id_s + (num_rasterizers_s - Y*NUM_BLOCKS) - bb_from_s) % num_rasterizers_s;
			int x_off = i - (bb_to.y - 1 - Y)*full_sets_per_line;
			X = bb_from.x + step_s / PARTIAL + x_off*NUM_BLOCKS;
		}
		else
		{
			bb_from_s = bb_from_s + (full_sets_per_line*NUM_BLOCKS*PARTIAL);

			int step = (bb_from_s - rasterizer_id_s + (NUM_BLOCKS + 1)*NUM_BLOCKS - 1) / NUM_BLOCKS - NUM_BLOCKS;

			int part_start_column_s = rasterizer_id_s + step*NUM_BLOCKS;
			int part_start_row = (-step + num_bins_y * PARTIAL + 1);
			int possible_per_rep = ((bb_max_s - part_start_column_s + NUM_BLOCKS) / NUM_BLOCKS);

			int lines_small = part_start_row - bb_to.y;
			int seq_reps_small = lines_small / PARTIAL;
			int found_rem_small = (seq_reps_small*possible_per_rep + min(possible_per_rep, lines_small - seq_reps_small*PARTIAL));

			int higher_id = found_rem_small + (i - found_full);

			int rel_seq = higher_id / possible_per_rep;
			int found_in_seqs = rel_seq*possible_per_rep;
			int base_y = part_start_row - rel_seq*PARTIAL;

			int id_in_seq = higher_id - found_in_seqs;

			Y = base_y - id_in_seq - 1;
			X = (part_start_column_s + id_in_seq*NUM_BLOCKS) / PARTIAL;
		}

		return make_int2(X, Y);
	}


	template <typename F>
	__device__
	//static unsigned int traverseRasterizers(int2 start, int2 end, F f)
	static unsigned int traverseRasterizers(int2 b_from, int2 b_end, F f)
	{
		//maybe rename later
		const int2 b_to = { b_end.x + 1, b_end.y + 1 };
		//

		const int rasterizer_id = rasterizer();

		const int h = b_to.y - b_from.y;

		int shift = b_from.y * NUM_BLOCKS;
		int def_shift = (shift + PARTIAL - 1) / PARTIAL;
		int def_shift_s = PARTIAL*def_shift;
		int rt = (b_from.x + def_shift) % NUM_BLOCKS;

		int startx = b_from.x, endx = b_from.x + NUM_BLOCKS;
		int start = startx, end = min(endx, b_to.x);

		unsigned int first_element = rt, element, found = 0;

		for (int row = 0; row < h && start < end; row++)
		{
			element = first_element + (start - b_from.x);

			for (int i = start; i < end; i++)
			{
				element = min(element, element - NUM_BLOCKS);
				f(element);
				element++;
				found++;
			}

			startx = end - def_shift_per_line;
			endx = endx - def_shift_per_line;
			first_element = first_element + def_shift_per_line;

			shift = shift + NUM_BLOCKS;
			def_shift_s = def_shift_s + def_shift_per_line*PARTIAL;
			int shift_compare = (def_shift_s - PARTIAL);

			if (shift <= shift_compare)
			{
				def_shift_s = shift_compare;
				startx = startx + 1;
				endx = endx + 1;
				first_element = first_element - 1;
			}

			start = max(startx, b_from.x);
			end = min(endx, b_to.x);
		}

		return found;
	}
};

template < int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId>
class PatternTileSpace <PATTERNTECHNIQUE::OFFSET_SHIFT, NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
	: public TileSpace<NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
{
	static constexpr int PARTIAL = OFFSET_PARAMETER;
	static constexpr int SD = (NUM_BLOCKS + 1) / PARTIAL;
	static constexpr int LD = SD + (NUM_BLOCKS + 1) - (SD * PARTIAL);
	static constexpr int SD_LD_DIFF = LD - SD;
	static constexpr int LD_SD_DIFF = 2 * SD - LD;
	static constexpr int LINES_PER_REP = ((PARTIAL * SD) + SD_LD_DIFF) * PARTIAL - PARTIAL;
	static constexpr int LINES_PER_BLOCK = SD * PARTIAL - 1;
	static constexpr int LINES_PER_BIG_BLOCK = LD * PARTIAL - 1;
	static constexpr int LINES_PER_INTRO_BLOCK = SD_LD_DIFF*PARTIAL - 1;
	static constexpr int BIG = 10000;
	static constexpr int BIG_NUM_R = NUM_BLOCKS * BIG;
	static constexpr int BIG_NUM_SD = SD * BIG;

	static constexpr int num_rasterizers_s = NUM_BLOCKS*PARTIAL;
	static constexpr int def_shift_per_line = (NUM_BLOCKS + PARTIAL - 1) / PARTIAL;

	struct Phase
	{
		int in_block;
		int in_line;
		int left;
	};

	struct PhaseState
	{
		Phase phase;
		bool intro;
		int lines;
		int sum;
	};

public:

	__inline__ __device__ static int firstInLine(int x, int y, int id)
	{
		int l_id = rrasterizer(math::int2(x, y));
		return x + (id - l_id) + (l_id > id ? NUM_BLOCKS : 0);
	}

	__inline__ __device__ static int firstZeroBelow(math::int2 point, int id)
	{
		int shift = ((BIG * NUM_BLOCKS) + (id - point.x)) % NUM_BLOCKS;
		int y = -(shift - SD) * PARTIAL;
		if (y > point.y)
		{	y -= LINES_PER_REP;	}
		return y;
	}

	__inline__ __device__ static int cap(PhaseState& state)
	{
		if (state.intro)
		{
			int sum = 0;
			int diags_seen = state.lines / PARTIAL;
			int l = diags_seen;
			int limit = min(l, state.phase.left);

			sum += limit * state.phase.in_line;
			l -= limit;
			limit = min(SD, l);
			sum += limit * max(0, state.phase.in_line - 1);
			l -= limit;
			sum += l * max(0, state.phase.in_line - 2);

			int max_for_line = max(0, state.phase.in_line - (diags_seen < state.phase.left ? 0 : 1) - (diags_seen < state.phase.left + SD ? 0 : 1));
			int lines_seen = state.lines - diags_seen * PARTIAL;

			sum += min(lines_seen, max_for_line);
			return state.sum - sum;
		}
		else
		{
			int blox = (state.lines - 1) / LINES_PER_BLOCK;
			state.lines -= blox * LINES_PER_BLOCK;

			int diags_seen = state.lines / PARTIAL;
			int in_diags = min(state.phase.left, diags_seen) * state.phase.in_line + max(0, diags_seen - state.phase.left) * (state.phase.in_line - 1);
			int lines_seen = state.lines - diags_seen * PARTIAL;
			int in_lines = min(lines_seen, state.phase.in_line - (diags_seen < state.phase.left ? 0 : 1));
			return state.sum - (blox * state.phase.in_block + in_diags + in_lines);
		}
	}

	__inline__ __device__ static void computePhases(int dim, Phase& small_phase, Phase& big_phase, Phase& intro_phase, Phase& outro_phase, int& blocks_w_small, int& blocks_w_big, int& outro_exists)
	{
		int smalls_exist = (dim > SD ? 1 : 0);
		small_phase.in_line = min(PARTIAL, (dim + SD - 1) / SD);
		small_phase.left = dim - (small_phase.in_line - 1) * SD;
		small_phase.in_block = min(SD, small_phase.left) * small_phase.in_line + max(0, min(dim, SD) - small_phase.left) * (small_phase.in_line - 1);

		int bigs_exist = (dim > LD ? 1 : 0);
		big_phase.in_line = max(1, small_phase.in_line - (SD_LD_DIFF < small_phase.left ? 0 : 1));
		big_phase.left = !bigs_exist * dim + bigs_exist * (dim - SD * (big_phase.in_line - 1) - SD_LD_DIFF);
		big_phase.in_block = big_phase.left * big_phase.in_line + (SD - big_phase.left) * (big_phase.in_line - 1);

		blocks_w_big = bigs_exist * (big_phase.in_line - 1);
		blocks_w_small = max(1, PARTIAL - blocks_w_big - 1);

		intro_phase.in_line = small_phase.in_line - (small_phase.left < SD ? 1 : 0);
		intro_phase.left = smalls_exist * min(SD, dim - SD * intro_phase.in_line);
		intro_phase.in_block = min(SD_LD_DIFF, intro_phase.left) * intro_phase.in_line + max(0, SD_LD_DIFF - intro_phase.left) * max(0, intro_phase.in_line - 1);

		outro_exists = (PARTIAL - blocks_w_big > 1 ? 1 : 0);
		outro_phase.in_line = big_phase.in_line;
		outro_phase.left = !smalls_exist * dim + smalls_exist * min(SD, dim - SD * (outro_phase.in_line - 1));
		outro_phase.in_block = outro_exists * (outro_phase.left * outro_phase.in_line + (SD - outro_phase.left) * (outro_phase.in_line - 1));
	}

	__inline__ __device__ static void updateState(bool condition, PhaseState& state, int max_blocks, Phase& phase, bool intro, int lines, int value)
	{
		if (state.sum >= 0)
		{
			if (condition)
			{
				state.phase = phase;
				state.intro = intro;
				state.lines = lines;
				state.sum = -value - 1;
			}
			else
			{
				state.sum += max_blocks * phase.in_block;
			}
		}
	}

	__inline__ __device__ static void findStartAndFinish(PhaseState& start, PhaseState& end, int max_blocks, Phase& phase, int bb_from_y, int bb_end_y, int& limit_y, int y_add, bool intro = false)
	{
		updateState(bb_from_y <= (limit_y + y_add), start, max_blocks, phase, intro, bb_from_y - limit_y, start.sum);
		updateState(bb_end_y <= (limit_y + y_add), end, max_blocks, phase, intro, bb_end_y - limit_y, end.sum);
		limit_y += y_add;
	}


	__inline__ __device__ static int2 findSumIndexInPhase(PhaseState& state, int bb_x, int bin_id)
	{
		int id = (-state.sum) - 1;
		int blox = (state.phase.in_block == 0 ? 0 : id / state.phase.in_block);
		int rem = (id - (blox*state.phase.in_block));
		int t = 0, diags = 0;

		if (state.phase.in_line > 0) //full diags
		{
			t = min(state.phase.left, rem / state.phase.in_line);
			diags += t;
			rem -= t * state.phase.in_line;

			if (t == state.phase.left && --state.phase.in_line) //not so full diags
			{
				t = min(SD, rem / state.phase.in_line);
				diags += t;
				rem -= t * state.phase.in_line;

				if (t == SD && --state.phase.in_line) //less full diags
				{
					t = rem / state.phase.in_line;
					diags += t;
					rem -= t * state.phase.in_line;
				}
			}
		}

		int y = state.lines + (blox * SD + diags) * PARTIAL - blox + rem;
		return make_int2(firstInLine(bb_x, y, bin_id), y);
	}

	__inline__ __device__ static void findStart(PhaseState& start, int max_blocks, Phase& phase, int bb_from_y, int& limit_y, int y_add, bool intro = false)
	{
		updateState(bb_from_y <= (limit_y + y_add), start, max_blocks, phase, intro, bb_from_y - limit_y, start.sum);
		limit_y += y_add;
	}

	__inline__ __device__ static void findIndexAfterStart(PhaseState& target, int max_blocks, Phase& phase, int i, int& limit_y, int y_add, bool intro = false)
	{
		updateState(i < (target.sum + max_blocks * phase.in_block), target, max_blocks, phase, intro, target.lines + limit_y, i - target.sum);
		limit_y += y_add;
	}

	static __forceinline__ __device__ int OIB(int a, int b)
	{
		//return (a > b ? 1 : 0);
		return !max(0, b - a + 1);
	}

	static __forceinline__ __device__ int DIVUP(int a, int b)
	{
		return (a + b - 1) / b;
	}

	__inline__ __device__ static int rrasterizer(math::int2 point)
	{
		int shift = point.y / PARTIAL;
		int in_line = (point.y - (shift*PARTIAL));
		int rem = in_line * SD;
		int id = ((BIG * NUM_BLOCKS) + (point.x - shift - rem)) % NUM_BLOCKS;
		return id;
	}

	__device__ static int numHitBinsForMyRasterizer(const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		const int id = TileSpace::rasterizer();

		math::int2 dim = bb_end - bb_from;

		if (dim.x <= 2 && dim.y <= 2)
		{
			int c_id;
			c_id = rrasterizer(bb_from);
			if (c_id == id)
			{	return 1;	}

			int horiz = (bb_from.x != bb_end.x - 1);
			int verti = (bb_from.y != bb_end.y - 1);

			if (horiz && (c_id + 1 == id || c_id + 1 == id + NUM_BLOCKS))
			{	return 1;	}

			if (verti)
			{
				math::int2 t_pos = math::int2(from.x, end.y);
				c_id = rrasterizer(t_pos);
				if (c_id == id)
				{	return 1;	}
				if (horiz && (c_id + 1 == id || c_id + 1 == id + NUM_BLOCKS))
				{	return 1;	}
			}
			return 0;
		}

		int full = dim.x / NUM_BLOCKS;
		int sum = full * (bb_end.y - bb_from.y);

		bb_from.x = bb_from.x + full * NUM_BLOCKS;
		dim.x = bb_end.x - bb_from.x;

		if (dim.x > 0)
		{
			Phase small_phase, big_phase, intro_phase, outro_phase;
			int blocks_w_small, blocks_w_big, outro_exists;
			computePhases(dim.x, small_phase, big_phase, intro_phase, outro_phase, blocks_w_small, blocks_w_big, outro_exists);

			int blocks_per_rep = intro_phase.in_block + outro_phase.in_block + blocks_w_small * small_phase.in_block + blocks_w_big * big_phase.in_block;
			int start_y = firstZeroBelow(bb_from, id);

			PhaseState start;
			bb_from.y = bb_from.y - start_y;
			start.sum = (bb_from.y < LINES_PER_REP ? 0 : blocks_per_rep);
			bb_from.y -= (bb_from.y < LINES_PER_REP ? 0 : LINES_PER_REP);

			PhaseState end;
			bb_end.y = bb_end.y - start_y;
			end.sum = (bb_end.y < LINES_PER_REP ? 0 : blocks_per_rep);
			bb_end.y -= (bb_end.y < LINES_PER_REP ? 0 : LINES_PER_REP);

			int limit_y = 0;
			findStartAndFinish(start, end, 1, intro_phase, bb_from.y, bb_end.y, limit_y, LINES_PER_INTRO_BLOCK, true); //Intro
			findStartAndFinish(start, end, blocks_w_big, big_phase, bb_from.y, bb_end.y, limit_y, blocks_w_big * LINES_PER_BLOCK); //Bigs
			if (outro_exists)
			{	findStartAndFinish(start, end, 1, outro_phase, bb_from.y, bb_end.y, limit_y, LINES_PER_BLOCK);	} //Outro
			findStartAndFinish(start, end, blocks_w_small, small_phase, bb_from.y, bb_end.y, limit_y, blocks_w_small * LINES_PER_BLOCK); //Smalls

			sum += cap(start) - cap(end);
		}
		return sum;
	}

	__device__ static int2 getHitBinForMyRasterizer(int i, const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		const int bin_id = TileSpace::rasterizer();

		math::int2 dim = bb_end - bb_from;

		if (dim.x <= 2 && dim.y <= 2)
		{
			int id;
			id = rrasterizer(bb_from);
			if (id == bin_id)
			{	return make_int2(from.x, from.y);	}

			if (id + 1 == bin_id || id + 1 == bin_id + NUM_BLOCKS)
			{	return make_int2(from.x + 1, bb_from.y);	}
			else
			{
				math::int2 t_pos = math::int2(from.x, end.y);
				id = rrasterizer(t_pos);
				if (id == bin_id)
				{	return make_int2(t_pos.x, t_pos.y);	}
				return make_int2(end.x, end.y);
			}
		}

		int full = dim.x / NUM_BLOCKS;
		int in_full_lines = full * (bb_end.y - bb_from.y);

		if (i < in_full_lines)
		{
			int y_l = i / full;
			int x_l = i - (y_l * full);
			int y = bb_from.y + y_l;
			int x = firstInLine(bb_from.x, y, bin_id) + x_l * NUM_BLOCKS;
			return make_int2(x, y);
		}
		else
		{
			bb_from.x = bb_from.x + full * NUM_BLOCKS;
			dim.x = bb_end.x - bb_from.x;

			Phase small_phase, big_phase, intro_phase, outro_phase;
			int blocks_w_small, blocks_w_big, outro_exists;
			computePhases(dim.x, small_phase, big_phase, intro_phase, outro_phase, blocks_w_small, blocks_w_big, outro_exists);

			int blocks_per_rep = intro_phase.in_block + outro_phase.in_block + blocks_w_small * small_phase.in_block + blocks_w_big * big_phase.in_block;

			PhaseState start, target;
			start.sum = target.sum = 0;
			target.lines = firstZeroBelow(bb_from, bin_id); //bring target lines directly to screen space

			bb_from.y = bb_from.y - target.lines;
			if (bb_from.y >= LINES_PER_REP)
			{
				start.sum = blocks_per_rep;
				bb_from.y -= LINES_PER_REP;
			}

			int limit_y = 0;
			findStart(start, 1, intro_phase, bb_from.y, limit_y, LINES_PER_INTRO_BLOCK, true); //Intro
			findStart(start, blocks_w_big, big_phase, bb_from.y, limit_y, blocks_w_big * LINES_PER_BLOCK); //Bigs
			if (outro_exists)
			{	findStart(start, 1, outro_phase, bb_from.y, limit_y, LINES_PER_BLOCK);	} //Outro
			findStart(start, blocks_w_small, small_phase, bb_from.y, limit_y, blocks_w_small * LINES_PER_BLOCK); //Smalls

			int relative_i = -cap(start) - 1 + (i - in_full_lines);
			if (relative_i >= blocks_per_rep)
			{
				target.sum = blocks_per_rep;
				target.lines += LINES_PER_REP;
			}

			limit_y = 0;
			findIndexAfterStart(target, 1, intro_phase, relative_i, limit_y, LINES_PER_INTRO_BLOCK, true);
			findIndexAfterStart(target, blocks_w_big, big_phase, relative_i, limit_y, blocks_w_big * LINES_PER_BLOCK); //Bigs
			if (outro_exists)
			{	findIndexAfterStart(target, 1, outro_phase, relative_i, limit_y, LINES_PER_BLOCK);	}
			findIndexAfterStart(target, blocks_w_small, small_phase, relative_i, limit_y, blocks_w_small * LINES_PER_BLOCK); //Smalls

			return findSumIndexInPhase(target, bb_from.x, bin_id);
		}
	}

	template <typename F>
	__device__ static unsigned int traverseRasterizers(int2 from, int2 end, F f)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		math::int2 dim = bb_end - bb_from;

		int lower_diags = bb_from.y / PARTIAL;
		int upper_diags = lower_diags + 1;
		int end_diags = bb_end.y / PARTIAL;
		int fulls = max(0, end_diags - upper_diags);

		if ((dim.x >= NUM_BLOCKS) || ((dim.x >= SD) && (PARTIAL * SD + fulls > NUM_BLOCKS)))
		{
			for (int i = 0; i < NUM_BLOCKS; i++)
			{	f(i);	}
			return NUM_BLOCKS;
		}
		else
		{
			int lower_y = lower_diags * PARTIAL;
			int upper_y = end_diags * PARTIAL;

			uint64_t in_line = (1ULL << dim.x) - 1;
			uint64_t combined = 0x0;

			if (fulls > 0)
			{
				for (int shift = 1; shift <= PARTIAL*SD; shift += SD)
				{	combined |= (in_line << shift) | (in_line >> (NUM_BLOCKS - shift));	}
				for (int shift = 1; shift <= min(fulls - 1, LD); shift++)
				{	combined |= (combined << 1) | (combined >> (NUM_BLOCKS - 1));	}
			}

			if (end_diags != lower_diags)
			{
				for (int i = 0, shift = fulls + 1; i < bb_end.y - upper_y; i++, shift += SD)
				{	combined |= (in_line << shift) | (in_line >> (NUM_BLOCKS - shift));	}
			}

			if (lower_diags != upper_diags)
			{
				for (int i = (bb_from.y - lower_y), shift = (bb_from.y - lower_y) * SD; (i < PARTIAL) && (i + lower_y < bb_end.y); i++, shift += SD)
				{	combined |= (in_line << shift) | (in_line >> (NUM_BLOCKS - shift));	}
			}

			combined &= (1ULL << NUM_BLOCKS) - 1;

			int seen = 0;
			int reference = rrasterizer(math::int2(end.x, lower_y));
			while (int current_id = __ffsll(combined))
			{
				combined -= (1ULL << current_id - 1);
				current_id = reference - (current_id - 1);
				current_id += (current_id < 0 ? NUM_BLOCKS : 0);

				f(current_id);
				seen++;
			}
			return seen;
		}
	}
};


template < int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId>
class PatternTileSpace <PATTERNTECHNIQUE::OFFSET_SHIFT_SLIM, NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
    : public TileSpace<NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
{
    static constexpr int PARTIAL = OFFSET_PARAMETER;
    static constexpr int SD = (NUM_BLOCKS + 1) / PARTIAL;
    static constexpr int LD = SD + (NUM_BLOCKS + 1) - (SD * PARTIAL);
    static constexpr int SD_LD_DIFF = LD - SD;
    static constexpr int LD_SD_DIFF = 2 * SD - LD;
    static constexpr int LINES_PER_REP = ((PARTIAL * SD) + SD_LD_DIFF) * PARTIAL - PARTIAL;
    static constexpr int LINES_PER_BLOCK = SD * PARTIAL - 1;
    static constexpr int LINES_PER_BIG_BLOCK = LD * PARTIAL - 1;
    static constexpr int LINES_PER_INTRO_BLOCK = SD_LD_DIFF*PARTIAL - 1;
    static constexpr int BIG = 10000;
    static constexpr int BIG_NUM_R = NUM_BLOCKS * BIG;
    static constexpr int BIG_NUM_SD = SD * BIG;

    static constexpr int num_rasterizers_s = NUM_BLOCKS*PARTIAL;
    static constexpr int def_shift_per_line = (NUM_BLOCKS + PARTIAL - 1) / PARTIAL;

public:

    static __forceinline__ __device__ int OIB(int a, int b)
    {
        //return (a > b ? 1 : 0);
        return !max(0, b - a + 1);
    }

    static __forceinline__ __device__ int DIVUP(int a, int b)
    {
        return (a + b - 1) / b;
    }

    __inline__ __device__ static int rrasterizer(math::int2 point)
    {
        int shift = point.y / PARTIAL;
        int in_line = (point.y - (shift*PARTIAL));
        int rem = in_line * SD;
        int id = ((BIG * NUM_BLOCKS) + (point.x - shift - rem)) % NUM_BLOCKS;
        return id;
    }

    __device__ static int numHitBinsForMyRasterizer(const int2 from, const int2 end)
    {
        ////
        math::int2 bb_from(from.x, from.y);
        math::int2 bb_end(end.x + 1, end.y + 1);
        ////

        int id = rasterizer();

        int dimx = bb_end.x - bb_from.x;
        int dimy = bb_end.y - bb_from.y;

        int full = dimx / NUM_BLOCKS;
        int sum = full * (bb_end.y - bb_from.y);

        bb_from.x += full * NUM_BLOCKS;
        dimx = bb_end.x - bb_from.x;

        int shift = bb_from.y / PARTIAL;
        int in_line = (bb_from.y - (shift*PARTIAL));
        int step = ((BIG_NUM_R)+id - (bb_from.x - shift - in_line * SD)) % NUM_BLOCKS;
        int shifts_left = step / SD;
        shifts_left -= OIB(shifts_left, in_line) & OIB(shifts_left * SD + SD_LD_DIFF, step);

        int left_x = (bb_from.x + step) - (shifts_left * SD + (shifts_left > in_line) * SD_LD_DIFF);
        int till_zero = (PARTIAL - in_line) + shifts_left;
        int in = DIVUP(bb_end.x - left_x, SD);

        till_zero -= OIB(till_zero, PARTIAL) * PARTIAL;
        in -= OIB(in, till_zero) & OIB(in * SD - LD_SD_DIFF + 1, (bb_end.x - left_x));

        int blox = DIVUP(dimy + shifts_left, PARTIAL);
        int till_in = SD + !(till_zero - PARTIAL) * SD_LD_DIFF - (left_x - bb_from.x) - 1;
        int till_out = !!in * (bb_end.x - (left_x + (in - 1) * SD + OIB(in, till_zero) * SD_LD_DIFF)) + !in*(till_in + dimx);

        int one_on_top = OIB(blox, till_in) * OIB(till_in + dimx + 1, blox);
        int remaining_lines = (bb_from.y - shifts_left + blox * PARTIAL) - bb_end.y;
        int in_last_block = !!remaining_lines * (in - OIB(blox, till_out) + OIB(blox, till_in));
        int in_last_line = one_on_top * (!!in_last_block) + max(0, remaining_lines - (PARTIAL - in_last_block + one_on_top));

        sum += blox * in - max(0, blox - till_out) + max(0, blox - till_in);
        sum -= min(in, shifts_left) + in_last_line;
        return sum;
    }

    __device__ static int2 getHitBinForMyRasterizer(int i, const int2 from, const int2 end)
    {
        //////
        math::int2 bb_from(from.x, from.y);
        math::int2 bb_end(end.x + 1, end.y + 1);
        int2 res;
        //////

        int id = rasterizer();
        int dimx = bb_end.x - bb_from.x;
        int dimy = bb_end.y - bb_from.y;
        int full = dimx / NUM_BLOCKS;
        int in_full_lines = full * (bb_end.y - bb_from.y);

        if (i < in_full_lines)
        {
            int y_l = i / full;
            int x_l = i - (y_l * full);
            res.y = bb_from.y + y_l;
            int shifts = res.y / PARTIAL;
            int offsets = res.y - shifts * PARTIAL;
            int step = (id + shifts + offsets * SD - bb_from.x + BIG_NUM_R) % NUM_BLOCKS;
            res.x = bb_from.x + step + x_l * NUM_BLOCKS;
        }
        else
        {
            bb_from.x += full * NUM_BLOCKS;
            dimx = bb_end.x - bb_from.x;

            res.x = dimx;
            res.y = dimx;

            int shift = bb_from.y / PARTIAL;
            int in_line = (bb_from.y - (shift*PARTIAL));
            int step = (BIG_NUM_R + id - (bb_from.x - shift - in_line * SD)) % NUM_BLOCKS;
            int shifts_left = step / SD;
            shifts_left -= OIB(shifts_left, in_line) & OIB(shifts_left * SD + SD_LD_DIFF, step);

            int left_x = bb_from.x + step - (shifts_left * SD + (shifts_left > in_line) * SD_LD_DIFF);
            int till_zero = (PARTIAL - in_line) + shifts_left;
            int in = DIVUP(bb_end.x - left_x, SD);

            till_zero -= OIB(till_zero, PARTIAL) * PARTIAL;
            in -= OIB(in, till_zero) & OIB(in * SD - LD_SD_DIFF, end.x - left_x);
            i += min(in, shifts_left) - in_full_lines;

            int till_in = SD + !(till_zero - PARTIAL) * SD_LD_DIFF - (left_x - bb_from.x) - 1;
            int till_out = !!in * (bb_end.x - left_x - (in - 1) * SD - OIB(in, till_zero) * SD_LD_DIFF) + !in * (till_in + dimx);
            int v = min(till_in, till_out), block = 0, line = in;

            if (i >= v * line)
            {
                i -= v * line;
                block = v;
                line = in + !(till_in - block) - !(till_out - block);
                v = max(till_in, till_out) - v;
                if (i >= v * line)
                {
                    i -= v * line;
                    block += v;
                    line = in + !max(0, till_in - block) - !max(0, till_out - block);
                }
            }

            int b = i / line;
            i -= b * line;
            int top = !(i - line + 1) & OIB(block + 1, till_in);
            block += b;
            res.x = top * (bb_from.x - till_in + block) + !top * (left_x + block + i*SD + OIB(i + 1, till_zero) * SD_LD_DIFF);
            res.y = bb_from.y - shifts_left + block * PARTIAL + !top * i + top * (PARTIAL - 1);
        }
        return res;
    }

    template <typename F>
    __device__ static unsigned int traverseRasterizers(int2 from, int2 end, F f)
    {
        ////
        math::int2 bb_from(from.x, from.y);
        math::int2 bb_end(end.x + 1, end.y + 1);
        ////

        math::int2 dim = bb_end - bb_from;

        int lower_diags = bb_from.y / PARTIAL;
        int upper_diags = lower_diags + 1;
        int end_diags = bb_end.y / PARTIAL;
        int fulls = max(0, end_diags - upper_diags);

        if ((dim.x >= NUM_BLOCKS) || ((dim.x >= SD) && (PARTIAL * SD + fulls > NUM_BLOCKS)))
        {
            for (int i = 0; i < NUM_BLOCKS; i++)
            {
                f(i);
            }
            return NUM_BLOCKS;
        }
        else
        {
            int lower_y = lower_diags * PARTIAL;
            int upper_y = end_diags * PARTIAL;

            uint64_t in_line = (1ULL << dim.x) - 1;
            uint64_t combined = 0x0;

            if (fulls > 0)
            {
                for (int shift = 1; shift <= PARTIAL*SD; shift += SD)
                {
                    combined |= (in_line << shift) | (in_line >> (NUM_BLOCKS - shift));
                }
                for (int shift = 1; shift <= min(fulls - 1, LD); shift++)
                {
                    combined |= (combined << 1) | (combined >> (NUM_BLOCKS - 1));
                }
            }

            if (end_diags != lower_diags)
            {
                for (int i = 0, shift = fulls + 1; i < bb_end.y - upper_y; i++, shift += SD)
                {
                    combined |= (in_line << shift) | (in_line >> (NUM_BLOCKS - shift));
                }
            }

            if (lower_diags != upper_diags)
            {
                for (int i = (bb_from.y - lower_y), shift = (bb_from.y - lower_y) * SD; (i < PARTIAL) && (i + lower_y < bb_end.y); i++, shift += SD)
                {
                    combined |= (in_line << shift) | (in_line >> (NUM_BLOCKS - shift));
                }
            }

            combined &= (1ULL << NUM_BLOCKS) - 1;

            int seen = 0;
            int reference = rrasterizer(math::int2(end.x, lower_y));
            while (int current_id = __ffsll(combined))
            {
                combined -= (1ULL << current_id - 1);
                current_id = reference - (current_id - 1);
                current_id += (current_id < 0 ? NUM_BLOCKS : 0);

                f(current_id);
                seen++;
            }
            return seen;
        }
    }
};


template < int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId>
class PatternTileSpace <PATTERNTECHNIQUE::DIAGONAL_ITERATIVE, NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
	: public TileSpace<NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
{
public:
	__device__ static int numHitBinsForMyRasterizer(const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		int id = rasterizer();

		int sum = 0;
		for (int i = bb_from.y; i < bb_end.y; i++)
		{
			for (int j = bb_from.x; j < bb_end.x; j++)
			{
				if (id == rasterizer(make_int2(j, i)))
				{
					sum++;
				}
			}
		}
		return sum;
	}

	__device__ static int2 getHitBinForMyRasterizer(int i, const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		int id = rasterizer();

		int sum = 0;
		for (int y = bb_from.y; y < bb_end.y; y++)
		{
			for (int x = bb_from.x; x < bb_end.x; x++)
			{
				if (id == rasterizer(make_int2(x, y)))
				{
					sum++;
				}
				if (sum == i + 1)
				{
					return make_int2(x, y);
				}
			}
		}
	}

	template <typename F>
	__device__ static unsigned int traverseRasterizers(int2 from, int2 end, F f)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		math::int2 dim = bb_end - bb_from;

		uint64_t v = 0x0ULL;
		uint64_t one = 0x1ULL;

		for (int i = 0; i < dim.y; i++)
		{
			for (int j = 0; j < dim.x; j++)
			{
				int id = rasterizer(make_int2(bb_from.x + j, bb_from.y + i));
				v |= (one << id);
			}
		}

		int seen = 0;
		while (int current_id = __ffs(v))
		{
			v -= (one << (current_id - 1));
			f(current_id - 1);
			seen++;
		}
		return seen;
	}
};

template < int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId>
class PatternTileSpace <PATTERNTECHNIQUE::OFFSET_SHIFT_ITERATIVE, NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
	: public TileSpace<NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
{
	static constexpr int PARTIAL = OFFSET_PARAMETER;
	static constexpr int SD = (NUM_BLOCKS + 1) / PARTIAL;
	static constexpr int LD = SD + (NUM_BLOCKS + 1) - (SD * PARTIAL);
	static constexpr int SD_LD_DIFF = LD - SD;
	static constexpr int BIG = 10000;
	static constexpr int BIG_NUM_R = NUM_BLOCKS * BIG;

	__inline__ __device__ static int rrasterizer(math::int2 point)
	{
		int shift = point.y / PARTIAL;
		int in_line = (point.y - (shift*PARTIAL));
		int rem = in_line * SD;
		int id = ((BIG * NUM_BLOCKS) + (point.x - shift - rem)) % NUM_BLOCKS;
		return id;
	}

public:
	__device__ static int numHitBinsForMyRasterizer(const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		int id = rasterizer();

		int sum = 0;
		for (int i = bb_from.y; i < bb_end.y; i++)
		{
			for (int j = bb_from.x; j < bb_end.x; j++)
			{
				if (id == rrasterizer(math::int2(j, i)))
				{
					sum++;
				}
			}
		}
		return sum;
	}

	__device__ static int2 getHitBinForMyRasterizer(int i, const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		int id = rasterizer();

		int sum = 0;
		for (int y = bb_from.y; y < bb_end.y; y++)
		{
			for (int x = bb_from.x; x < bb_end.x; x++)
			{
				if (id == rrasterizer(math::int2(x, y)))
				{
					sum++;
				}
				if (sum == i + 1)
				{
					return make_int2(x, y);
				}
			}
		}
	}

	template <typename F>
	__device__ static unsigned int traverseRasterizers(int2 from, int2 end, F f)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		math::int2 dim = bb_end - bb_from;

		uint64_t v = 0x0ULL;
		uint64_t one = 0x1ULL;

		for (int i = 0; i < dim.y; i++)
		{
			for (int j = 0; j < dim.x; j++)
			{
				int id = rrasterizer(math::int2(bb_from.x + j, bb_from.y + i));
				v |= (one << id);
			}
		}

		int seen = 0;
		while (int current_id = __ffs(v))
		{
			v -= (one << (current_id - 1));
			f(current_id - 1);
			seen++;
		}
		return seen;
	}
};

template < int NUM_BLOCKS, int x_max, int y_max, int bin_size_x, int bin_size_y, int stamp_width, int stamp_height, class RasterizerId>
class PatternTileSpace <PATTERNTECHNIQUE::OFFSET_ITERATIVE, NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
	: public TileSpace<NUM_BLOCKS, x_max, y_max, bin_size_x, bin_size_y, stamp_width, stamp_height, RasterizerId>
{
	static constexpr int PARTIAL = OFFSET_PARAMETER;
	static constexpr int SD = (NUM_BLOCKS + 1) / PARTIAL;
	static constexpr int LD = SD + (NUM_BLOCKS + 1) - (SD * PARTIAL);
	static constexpr int SD_LD_DIFF = LD - SD;
	static constexpr int BIG = 10000;
	static constexpr int BIG_NUM_R = NUM_BLOCKS * BIG;

	__inline__ __device__ static int rrasterizer(math::int2 point)
	{
		int id = ((point.x*PARTIAL + point.y*NUM_BLOCKS) / PARTIAL) % NUM_BLOCKS;
		return id;
	}

public:
	__device__ static int numHitBinsForMyRasterizer(const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		int id = rasterizer();

		int sum = 0;
		for (int i = bb_from.y; i < bb_end.y; i++)
		{
			for (int j = bb_from.x; j < bb_end.x; j++)
			{
				if (id == rrasterizer(math::int2(j, i)))
				{
					sum++;
				}
			}
		}
		return sum;
	}

	__device__ static int2 getHitBinForMyRasterizer(int i, const int2 from, const int2 end)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		int id = rasterizer();

		int sum = 0;
		for (int y = bb_from.y; y < bb_end.y; y++)
		{
			for (int x = bb_from.x; x < bb_end.x; x++)
			{
				if (id == rrasterizer(math::int2(x, y)))
				{
					sum++;
				}
				if (sum == i + 1)
				{
					return make_int2(x, y);
				}
			}
		}
	}

	template <typename F>
	__device__ static unsigned int traverseRasterizers(int2 from, int2 end, F f)
	{
		////
		math::int2 bb_from(from.x, from.y);
		math::int2 bb_end(end.x + 1, end.y + 1);
		////

		math::int2 dim = bb_end - bb_from;

		uint64_t v = 0x0ULL;
		uint64_t one = 0x1ULL;

		for (int i = 0; i < dim.y; i++)
		{
			for (int j = 0; j < dim.x; j++)
			{
				int id = rrasterizer(math::int2(bb_from.x + j, bb_from.y + i));
				v |= (one << id);
			}
		}

		int seen = 0;
		while (int current_id = __ffs(v))
		{
			v -= (one << (current_id - 1));
			f(current_id - 1);
			seen++;
		}
		return seen;
	}
};

#endif  // INCLUDED_CURE_OFFSET_SHIFT_BIN_TILE_SPACE
