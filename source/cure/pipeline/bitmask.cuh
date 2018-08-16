


#ifndef INCLUDED_CURE_BITMASK
#define INCLUDED_CURE_BITMASK

#pragma once

#include <ptx_primitives.cuh>


template<int num_uints, int cols, int rows>
struct IBitMask;


template<typename T, int cols, int rows>
struct RowRepeater
{
	static constexpr T Repeater = (RowRepeater<T, cols, rows - 1>::Repeater << cols) | 1;
};

template<typename T, int cols>
struct RowRepeater<T, cols, 1>
{
	static constexpr T Repeater = 1;
};

template<typename T, int cols, int rows, int offset, int initoffset = 0>
struct RowRepeaterOffset
{
	static constexpr T Repeater = RowRepeaterOffset<T, cols, (rows - offset > 0 ? rows - offset : 0), offset, cols*offset + initoffset>::Repeater | (static_cast<T>(1u) << static_cast<T>(initoffset));
};

template<typename T, int cols, int offset, int initoffset >
struct RowRepeaterOffset<T, cols, 0, offset, initoffset>
{
	static constexpr T Repeater = 0;
};

template<int cols, int rows>
struct IBitMask<1, cols, rows>
{
	static constexpr int Cols = cols;
	static constexpr int Rows = rows;

	unsigned int mask;

	IBitMask() = default;

	__device__
	IBitMask(unsigned int mask)
		: mask(mask)
	{
	}

	__device__
	void setOn()
	{
		mask = 0xFFFFFFFFU;
	}
	__device__
	void unmarkBits(int from, int to)
	{
		int num = to - from;
		unsigned int tmask = (0xFFFFFFFFU >> (32 - num)) << from;
		mask = mask & ~tmask;
	}
	__device__
	void unmarkRow(int row, unsigned int rowmask)
	{
		mask = mask & (~(rowmask << (cols*row)));
	}

	__device__
	void unmark(const IBitMask& other)
	{
		mask = mask & (~other.mask);
	}

	__device__
	void unmarkStride(int begin, int end)
	{
		mask = mask & (~(((0x1U << (end - begin)) - 1) << begin));
	}

	__device__
	void andStride(int begin, int end)
	{
		mask = mask & (((0x1U << (end - begin)) - 1) << begin);
	}

	__device__
	void repeatRow(int begin, int end)
	{
		unsigned int row = ((0x1U << (end - begin)) - 1) << begin;
		mask = RowRepeater<unsigned int, cols, rows>::Repeater * row;
	}


	__device__
	bool isset(int col, int row) const
	{
		return 1U & (mask >> (cols*row + col));
	}
	__device__
	int count() const
	{
		return __popc(mask);
	}
	__device__
	int2 getBitCoordsWarp(int setBitId) const
	{
		return bitToCoord(getSetBitWarp(setBitId));
	}

	__device__
	int getSetBitWarp(int setBitId) const
	{
		// find the thread that has the right number of bits set
		return __ffs(__ballot_sync(~0U, __popc(mask & lanemask_lt()) == setBitId)) - 1;
	}

	__device__
	int2 getBitCoords(int setBitId) const
	{
		return bitToCoord(getSetBit(setBitId));
	}


	__device__
	int getSetBit(int setBitId) const
	{
		// find the nth set bit
		int invset = __popc(mask) - setBitId - 1;
		unsigned int p = 16;
#pragma unroll
		for (unsigned int offset = p / 2; offset > 0; offset /= 2)
			p = (__popc(mask >> p) <= invset) ? (p - offset) : (p + offset);
		p = (__popc(mask >> p) == invset) ? (p - 1) : p;
		return p;
	}

	__device__
	static int2 bitToCoord(int bit)
	{
		return make_int2(bit % cols, bit / cols);
	}

	__device__
	IBitMask shfl(int i, int Mod = WARP_SIZE) const
	{
		IBitMask other;
		other.mask = __shfl_sync(~0U, mask, i, Mod);
		return other;
	}

	__device__ bool overlap(IBitMask other) const
	{
		return (mask & other.mask) != 0U;
	}

	__device__
	static IBitMask Empty()
	{
		IBitMask e;
		e.mask = 0x00000000U;
		return e;
	}

	static constexpr unsigned int SecondRowMask = RowRepeaterOffset<unsigned int, cols, rows, 2, 0>::Repeater * ((1u << cols) - 1);
	static constexpr unsigned int Rights = RowRepeater<unsigned int, 2, cols*rows / 2>::Repeater;
	static constexpr unsigned int Lefts = Rights << 1u;

	//__device__
	//int countQuads() const
	//{
	//	//merge every second row
	//	unsigned int rowmerged = (mask & SecondRowMask) | ((mask >> cols) & SecondRowMask);
	//	////mask out the right ones and shift to the left
	//	//unsigned int expandedLeft = (rowmerged & Lefts) << 1u;
	//	////mask out the left ones and shift to the right
	//	//unsigned int expandedRight = (rowmerged & Rights) >> 1u;
	//	//unsigned int quadsDouble = expandedLeft | rowmerged | expandedRight;
	//	//return __popc(quadsDouble) / 2;

	//	unsigned int quadMask = (rowmerged & Rights) | ((rowmerged & Lefts) >> 1u)
	//}

	__device__
	IBitMask quadMask() const
	{
		unsigned int rowmerged = (mask & SecondRowMask) | ((mask >> cols) & SecondRowMask);
		unsigned int quadMask = (rowmerged & Rights) | ((rowmerged & Lefts) >> 1u);
		
		IBitMask res;
		res.mask = quadMask;
		return res;
	}


	__device__
	IBitMask operator ^=(const IBitMask& b)
	{
		mask ^= b.mask;
		return *this;
	}

	__device__
	IBitMask operator &=(const IBitMask& b)
	{
		mask &= b.mask;
		return *this;
	}

	__device__
	IBitMask operator |=(const IBitMask& b)
	{
		mask |= b.mask;
		return *this;
	}

	__device__
	IBitMask friend operator ~(const IBitMask& b)
	{
		return { ~b.mask };
	}

	__device__
	IBitMask friend operator ^(IBitMask a, const IBitMask& b)
	{
		return a ^= b;
	}

	__device__
	IBitMask friend operator &(IBitMask a, const IBitMask& b)
	{
		return a &= b;
	}

	__device__
	IBitMask friend operator |(IBitMask a, const IBitMask& b)
	{
		return a |= b;
	}
};




template<int cols, int rows>
struct IBitMask<2, cols, rows>
{
	static constexpr int Cols = cols;
	static constexpr int Rows = rows;

	unsigned long long mask;

	IBitMask() = default;

	__device__
	IBitMask(unsigned long long mask)
		: mask(mask)
	{
	}

	__device__
	void setOn()
	{
		mask = 0xFFFFFFFFFFFFFFFFULL;
	}

	__device__
	void unmarkBits(int from, int to)
	{
		int num = to - from;
		unsigned long long int tmask = (0xFFFFFFFFFFFFFFFFULL >> (64 - num)) << from;
		mask = mask & (~tmask);
	}
	__device__
	void unmarkRow(int row, unsigned long long int rowmask)
	{
		mask = mask & (~(rowmask << (cols*row)));
	}
	__device__
	void unmark(const IBitMask& other)
	{
		mask = mask & (~other.mask);
	}

	__device__
	void unmarkStride(int begin, int end)
	{
		mask = mask & (~(((0x1ULL << (end - begin)) - 1) << begin));
	}

	__device__
	void andStride(int begin, int end)
	{
		mask = mask & (((0x1ULL << (end - begin)) - 1) << begin);
	}

	__device__
	void repeatRow(int begin, int end)
	{
		unsigned int row = ((0x1U << (end - begin)) - 1) << begin;
		mask = RowRepeater<unsigned long long int, cols, rows>::Repeater * row;
	}

	__device__
	bool isset(int col, int row) const
	{
		return 1ULL & (mask >> (cols*row + col));
	}

	__device__
	int count() const
	{
		return __popcll(mask);
	}

	__device__
	int2 getBitCoordsWarp(int setBitId) const
	{
		return bitToCoord(getSetBitWarp(setBitId));
	}

	__device__
	int getSetBitWarp(int setBitId) const
	{
		// find the thread that has the right number of bits set
		unsigned int lower = mask & 0xFFFFFFFFU;
		unsigned int lowernum = __popc(lower);
		bool is_in_high = lowernum <= setBitId;
		unsigned int checkmask = is_in_high ? (mask >> 32U) : lower;
		setBitId = is_in_high ? setBitId - lowernum : setBitId;
		int fieldoffset = is_in_high ? 32 : 0;
		int offset = fieldoffset + __ffs(__ballot_sync(~0U, __popc(checkmask & lanemask_le()) == setBitId + 1)) - 1;
		return offset;
	}


	__device__
	int2 getBitCoords(int setBitId) const
	{
		return bitToCoord(getSetBit(setBitId));
	}


	__device__
	int getSetBit(int setBitId) const
	{
		// find the nth set bit
		int invset = __popcll(mask) - setBitId - 1;
		unsigned int p = 32;
#pragma unroll
		for (unsigned int offset = p / 2; offset > 0; offset /= 2)
			p = (__popcll(mask >> p) <= invset) ? (p - offset) : (p + offset);
		p = (__popcll(mask >> p) == invset) ? (p - 1) : p;
		return p;
	}

	__device__
	static int2 bitToCoord(int bit)
	{
		return make_int2(bit % cols, bit / cols);
	}

	__device__
	IBitMask shfl(int i, int Mod = WARP_SIZE) const
	{
		IBitMask other;
		other.mask = (static_cast<unsigned long long int>(__shfl_sync(~0U, static_cast<unsigned int>(mask >> 32), i, Mod)) << 32U) | 
			static_cast<unsigned long long int>(__shfl_sync(~0U, static_cast<unsigned int>(mask & 0xFFFFFFFFULL), i, Mod));
		return other;
	}

	__device__
	bool overlap(IBitMask other) const
	{
		return (mask & other.mask) != 0ULL;
	}

	__device__
	static IBitMask Empty()
	{
		IBitMask e;
		e.mask = 0x0000000000000000ULL;
		return e;
	}

	static constexpr unsigned long long int SecondRowMask = RowRepeaterOffset<unsigned long long int, cols, rows, 2, 0>::Repeater * ((1u << cols) - 1);
	static constexpr unsigned long long int Rights = RowRepeater<unsigned long long int, 2, cols*rows / 2>::Repeater;
	static constexpr unsigned long long int Lefts = Rights << 1u;

	__device__
	IBitMask quadMask() const
	{
		unsigned long long int rowmerged = (mask & SecondRowMask) | ((mask >> cols) & SecondRowMask);
		unsigned long long int quadMask = (rowmerged & Rights) | ((rowmerged & Lefts) >> 1u);

		IBitMask res;
		res.mask = quadMask;
		return res;
	}

		__device__
	IBitMask operator ^=(const IBitMask& b)
	{
		mask ^= b.mask;
		return *this;
	}

	__device__
	IBitMask operator &=(const IBitMask& b)
	{
		mask &= b.mask;
		return *this;
	}

	__device__
	IBitMask operator |=(const IBitMask& b)
	{
		mask |= b.mask;
		return *this;
	}

	__device__
	IBitMask friend operator ~(const IBitMask& b)
	{
		return { ~b.mask };
	}

	__device__
	IBitMask friend operator ^(IBitMask a, const IBitMask& b)
	{
		return a ^= b;
	}

	__device__
	IBitMask friend operator &(IBitMask a, const IBitMask& b)
	{
		return a &= b;
	}

	__device__
	IBitMask friend operator |(IBitMask a, const IBitMask& b)
	{
		return a |= b;
	}
};

template <class BinTileSpace>
class TileBitMask : public IBitMask<static_divup<BinTileSpace::TilesPerBinX*BinTileSpace::TilesPerBinY, 32>::value, BinTileSpace::TilesPerBinX, BinTileSpace::TilesPerBinY>
{ 
public:
	TileBitMask() = default;
	__device__ TileBitMask(const IBitMask<static_divup<BinTileSpace::TilesPerBinX*BinTileSpace::TilesPerBinY, 32>::value, BinTileSpace::TilesPerBinX, BinTileSpace::TilesPerBinY> & other)
	{
		mask = other.mask;
	}
	__device__
		bool operator == (const TileBitMask & other)
	{
		return mask == other.mask;
	}
};

template <class BinTileSpace>
class StampBitMask : public IBitMask<static_divup<BinTileSpace::StampsPerTileX*BinTileSpace::StampsPerTileY, 32>::value, BinTileSpace::StampsPerTileX, BinTileSpace::StampsPerTileY>
{
public:
	StampBitMask() = default;
	__device__ StampBitMask(const IBitMask<static_divup<BinTileSpace::StampsPerTileX*BinTileSpace::StampsPerTileY, 32>::value, BinTileSpace::StampsPerTileX, BinTileSpace::StampsPerTileY> & other)
	{
		mask = other.mask;
	}
	__device__
	bool operator == (const StampBitMask & other)
	{
		return mask == other.mask;
	}
};

#endif