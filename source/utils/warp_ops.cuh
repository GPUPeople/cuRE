


#ifndef INCLUDED_WARP_OPS
#define INCLUDED_WARP_OPS

#pragma once

#include "utils.h"


template<int GroupSize>
inline __device__ unsigned int GroupId() {
	return laneid() % GroupSize;
}

template<>
inline __device__ unsigned int GroupId<32>() {
	return laneid();
}

template<int GroupSize>
inline __device__ unsigned int GroupMask() {
	return (1u << (GroupId<GroupSize>() + 1)) - 1;
}
template<int GroupSize>
inline __device__ unsigned int GroupMaskLt() {
	return (1u << GroupId<GroupSize>()) - 1;
}


namespace detail {
	template<typename T>
	class WarpScanImpl {
	public:
		static __device__ void FindMax(T val, T& maxVal, int lane_id) {
			maxVal = val;
			int offset = 1;
			#pragma unroll
			for (int i = 0; i < 5; i++) {
				T v = __shfl_up_sync(~0U, maxVal, offset, 32);

				if (lane_id >= offset)
					maxVal = max(maxVal, v);

				offset *= 2;
			}
		}
		static __device__ void FindMin(T val, T& minVal, int lane_id) {
			minVal = val;
			int offset = 1;
			#pragma unroll
			for (int i = 0; i < 5; i++) {
				T v = __shfl_up_sync(~0U, minVal, offset, 32);

				if (lane_id >= offset)
					minVal = min(minVal, v);

				offset *= 2;
			}
		}

		static __device__ void InclusiveSum(T val, T& sum, int lane_id) {
			sum = val;
			int offset = 1;
			#pragma unroll
			for (int i = 0; i < 5; i++) {
				T v = __shfl_up_sync(~0U, sum, offset, 32);

				if (lane_id >= offset)
					sum += v;

				offset *= 2;
			}
		}

		static __device__ void InclusiveWeightedSum(T val, T& sum, float weight, int lane_id) {
			sum = val;
			int offset = 1;
			#pragma unroll
			for (int i = 0; i < 5; i++) {
				T v = __shfl_up_sync(~0U, sum, offset, 32);
				float chain_weight = weight;
				for (int j = 1; j < offset; j++) {
					chain_weight *= __shfl_sync(~0U, weight, lane_id - j);
				}
				if (lane_id >= offset) {
					//printf("lid: [%d], offset: [%d], cur_weight=%.3f, weighted=%.3f\n",  lane_id, offset, weight, chain_weight);
					sum += chain_weight * v;
				}

				offset *= 2;
			}
		}

		static __device__ void MaxLeft(T val, T& res, int lane_id) {
			res = val;
			int offset = 1;
			#pragma unroll
			for (int i = 0; i < 5; i++) {
				T v = __shfl_up_sync(~0U, res, offset, 32);
				res = max(v, res);
				offset *= 2;
			}
		}
	};
} // namespace detail.

template<typename T>
class WarpScan {
public:

	__device__ static void FindMax(T val, T& maxVal, int lane_id) {
		detail::WarpScanImpl<T>::FindMax(val, maxVal, lane_id);
	}

	__device__ static void FindMin(T val, T& minVal, int lane_id) {
		detail::WarpScanImpl<T>::FindMin(val, minVal, lane_id);
	}

	__device__ static void InclusiveSum(T val, T& sum, int lane_id) {
		detail::WarpScanImpl<T>::InclusiveSum(val, sum, lane_id);
	}

	__device__ static void InclusiveWeightedSum(T val, T& sum, float weight, int lane_id) {
		detail::WarpScanImpl<T>::InclusiveSum(val, sum, weight, lane_id);
	}

	__device__ static void MaxLeft(T val, T& res, int lane_id) {
		detail::WarpScanImpl<T>::MaxLeft(val, res, lane_id);
	}
};



template<typename T>
class GroupScan {
public:
	template<int GroupSize>
	static __device__ T InclusiveSum(T val, T& sum) {
		int lid = GroupId<GroupSize>();
		sum = val;
		int offset = 1;
		#pragma unroll
		for (int i = 1; i < GroupSize; i *= 2) {
			T v = __shfl_up_sync(~0U, sum, offset, GroupSize);

			if (lid >= offset)
				sum += v;

			offset *= 2;
		}
		return __shfl_sync(~0U, sum, GroupSize - 1, GroupSize);
	}

	template<int GroupSize>
	static __device__ T Sum(T val, int lid) {
		T sum = val;
		int offset = 1;
		#pragma unroll
		for (int i = 1; i < GroupSize; i *= 2) {
			T v = __shfl_up_sync(~0U, sum, offset, GroupSize);
			sum += v;
			offset *= 2;
		}
		return __shfl_sync(~0U, sum, GroupSize - 1, GroupSize);
	}

	template<int GroupSize>
	static __device__ T Sum(T val) {
		return Sum<GroupSize>(val, GroupId<GroupSize>());
	}

	template<int GroupSize>
	static __device__ T InclusiveWeightedSum(T val, T& sum, float weight) {
		int lid = GroupId<GroupSize>();
		sum = val;
		int offset = 1;
		#pragma unroll
		for (int i = 1; i < GroupSize; i *= 2) {
			T v = __shfl_up_sync(~0U, sum, offset, GroupSize);
			float chain_weight = weight;
			for (int j = 1; j < offset; j++) {
				chain_weight *= __shfl_sync(~0U, weight, lid - j, GroupSize);
			}
			if (lid >= offset) {
				//printf("lid: [%d], offset: [%d], cur_weight=%.3f, weighted=%.3f\n",  lane_id, offset, weight, chain_weight);
				sum += chain_weight * v;
			}

			offset *= 2;
		}
		return __shfl_sync(~0U, sum, GroupSize - 1, GroupSize);
	}
};

template<int GroupSize>
__device__ inline unsigned int ballotGroup(int val) {
	unsigned int res = __ballot_sync(~0U, val);
	int group = laneid() / GroupSize;
	return (res >> (group * GroupSize)) & ((1u << GroupSize) - 1u);
}

template<>
__device__ inline unsigned int ballotGroup<32>(int val) {
	return __ballot_sync(~0U, val);
}

template<int GroupSize>
__device__ inline int anyGroup(int val) {
	return (ballotGroup<GroupSize>(val)) != 0;
}

template<>
__device__ inline int anyGroup<32>(int val) {
	return __any_sync(~0U, val);
}

template<int GroupSize>
__device__ inline int allGroup(int val) {
	return ballotGroup<GroupSize>(val) == ((1u << GroupSize) - 1u);
}

template<>
__device__ inline int allGroup<32>(int val) {
	return __all_sync(~0U, val);
}


template<class T, bool Dir>
class SorterGroup
{
	template<int GroupSize>
	static __device__ inline void comp_and_exchange(int lid, int otherlid, T& v, int& id, bool dir)
	{
		// exchange values
		T other_v = __shfl_sync(~0U, v, otherlid, GroupSize);
		if ((v > other_v) == dir)
		{
			// swap
			v = other_v;
			id = __shfl_sync(~0U, id, otherlid, GroupSize);
		}
	}

public:
	template<int GroupSize>
	static __device__ inline int BitonicShflSortGroup(T& v)
	{
		int lid = GroupId<GroupSize>();
		return BitonicShflSortGroup(v, lid);
	}
	template<int GroupSize>
	static __device__ inline int BitonicShflSortGroup(T& v, int lid)
	{
		int dataid = lid;
		for (int size = 1; size < GroupSize / 2; size *= 2)
		{
			bool patchDir = Dir ^ (((lid / 2) & size) != 0);
			//bitonic merge
			for (int stride = size; stride > 0; stride /= 2)
			{
				bool strideDir = (lid & stride) != 0;
				comp_and_exchange<GroupSize>(lid, lid + (strideDir ? -stride : stride), v, dataid, patchDir ^ strideDir);
			}
		}

		//final merge
		for (uint stride = GroupSize / 2; stride > 0; stride /= 2)
		{
			bool strideDir = (lid & stride) != 0;
			comp_and_exchange<GroupSize>(lid, lid + (strideDir ? -stride : stride), v, dataid, Dir ^ strideDir);
		}
		return dataid;
	}
};

#endif  // INCLUDED_WARP_OPS
