


#ifndef INCLUDED_CURE_INDEX_QUEUE
#define INCLUDED_CURE_INDEX_QUEUE

#pragma once

#include <ptx_primitives.cuh>

namespace IndexQueueAccessControl
{
	struct AtomicCheckedAbortOnOverflow
	{
		template<unsigned int SIZE, class T>
		__device__
		static int enqueue(const T& element, int& count, unsigned int& back, T* indices, T UNUSED)
		{
			int fill = atomicAdd(&count, 1);
			if (fill < static_cast<int>(SIZE-1))
			{
				unsigned int pos = atomicInc(&back, SIZE - 1U);
				while (atomicCAS(indices + pos, UNUSED, element) != UNUSED)
					__threadfence();
				return fill;
			}
			else
				__trap();
		}

		template<unsigned int SIZE, class T>
		__device__
		static void read(T& localElement, T* readElement, T UNUSED)
		{
			while ((localElement = atomicExch(readElement, UNUSED)) == UNUSED)
				__threadfence();
		}
	};

	struct AtomicCheckedWaitOnOverflow
	{
		template<unsigned int SIZE, class T>
		__device__
		static int enqueue(const T& element, int& count, unsigned int& back, T* indices, T UNUSED)
		{
			int fill = atomicAdd(&count, 1);
			unsigned int pos = atomicInc(&back, SIZE - 1U);
			while (atomicCAS(indices + pos, UNUSED, element) != UNUSED)
				__threadfence();
			return fill;
		}

		template<unsigned int SIZE, class T>
		__device__
		static void read(T& localElement, T* readElement, T UNUSED)
		{
			while ((localElement = atomicExch(readElement, UNUSED)) == UNUSED)
				__threadfence();
		}
	};

	struct NonAtomicCheckedAbortOnOverflow
	{
		template<unsigned int SIZE, class T>
		__device__
		static int enqueue(const T& element, int& count, unsigned int& back, T* indices, T UNUSED)
		{
			int fill = atomicAdd(&count, 1);
			if (fill < static_cast<int>(SIZE-1))
			{
				unsigned int pos = atomicInc(&back, SIZE - 1U);
				while (ldg_cg(indices + pos) != UNUSED)
					__threadfence();
				stg_cg(indices + pos, element);
			}
			else
				__trap();

			return fill;
		}

		template<unsigned int SIZE, class T>
		__device__
		static void read(T& localElement, T* readElement, T UNUSED)
		{
			while ((localElement = ldg_cg(readElement)) == UNUSED)
				__threadfence();
			stg_cg(readElement, UNUSED);
		}
	};

	
	struct NonAtomicCheckedWaitOnOverflow
	{
		template<unsigned int SIZE, class T>
		__device__
		static int enqueue(const T& element, int& count, unsigned int& back, T* indices, T UNUSED)
		{
			int fill = atomicAdd(&count, 1);
			//while (fill >= static_cast<int>(SIZE))
			//	fill = ldg_cg(&count);

			unsigned int pos = atomicInc(&back, SIZE - 1U);
			while (ldg_cg(indices + pos) != UNUSED)
				__threadfence();
			stg_cg(indices + pos, element);
			return fill;
		}

		template<unsigned int SIZE, class T>
		__device__
		static void read(T& localElement, T* readElement, T UNUSED)
		{
			while ((localElement = ldg_cg(readElement)) == UNUSED)
				__threadfence();
			stg_cg(readElement, UNUSED);
		}
	};



	template <bool AtomicAccess, bool AbortOnOverflow>
	class EnumAccessControl;

	template <>
	class EnumAccessControl<true, true> : public AtomicCheckedAbortOnOverflow{};

	template <>
	class EnumAccessControl<true, false> : public AtomicCheckedWaitOnOverflow{};

	template <>
	class EnumAccessControl<false, true> : public NonAtomicCheckedAbortOnOverflow{};

	template <>
	class EnumAccessControl<false, false> : public NonAtomicCheckedWaitOnOverflow{};


}



template <unsigned int NUMQUEUES, unsigned int SIZE, typename T = unsigned int, class AccessControl = IndexQueueAccessControl::AtomicCheckedAbortOnOverflow, T UNUSED = static_cast<T>(-1), bool TRACK_FILL_LEVEL = false>
class MultiIndexQueue
{
	static constexpr int InternalCounterSize = (NUMQUEUES + 1023U) / 1024U * 1024U;
	//static constexpr T UNUSED = static_cast<T>(-1);

	int count[InternalCounterSize];
	unsigned int front[InternalCounterSize];
	unsigned int back[InternalCounterSize];

	int max_fill_level[InternalCounterSize];

	T indices[NUMQUEUES][SIZE];

public:
	typedef T Type;

	__device__
	void init()
	{
		int linid = blockIdx.x * blockDim.x + threadIdx.x;
		if (linid < InternalCounterSize)
		{
			count[linid] = 0;
			front[linid] = back[linid] = 0U;

			if (TRACK_FILL_LEVEL)
				max_fill_level[linid] = 0;
		}
		for (int q = 0; q < NUMQUEUES; ++q)
			for (int i = linid; i < SIZE; i += blockDim.x * gridDim.x)
				indices[q][i] = UNUSED;
	}

	__device__
	void writeMaxFillLevel(int* dest)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUMQUEUES; i += blockDim.x * gridDim.x)
			dest[i] = max_fill_level[i];
	}


	__device__
	void enqueue(int q, T i)
	{
		int fill_level = AccessControl:: template enqueue<SIZE,T>(i, count[q], back[q], indices[q], UNUSED);
		if (TRACK_FILL_LEVEL)
			atomicMax(&max_fill_level[q], fill_level);
	}

	__device__
	int singleThreadReserveRead(int q, int num)
	{
		int readable = atomicSub(&count[q], num);
		if (readable < num)
		{
			int putback = min(num - readable, num);
			atomicAdd(&count[q], putback);
			num = num - putback;
		}
		return num;
	}

	__device__
	int singleThreadTake(int q, int num)
	{
		return atomicAdd(&front[q], num);
	}

	__device__
	void multiThreadRead(int q, T* localElement, int tid, int offset)
	{
		int pos = (offset + tid) % SIZE;
		T el;
		AccessControl:: template read<SIZE, T>(el, indices[q] + pos, UNUSED);
		*localElement = el;
	}

	__device__
	int dequeueBlock(int q, T* localElement, int num)
	{
		__shared__ int take, offset;
		if (threadIdx.x == 0)
		{
			int ttake = singleThreadReserveRead(q, num);
			offset = singleThreadTake(q, ttake);
			take = ttake;
		}
		__syncthreads();
		if (threadIdx.x < take)
			multiThreadRead(q, localElement, threadIdx.x, offset);
		return take;
	}

	__device__
	int dequeueBlockUnsave(int q, T* localElement, int num)
	{
		__shared__ int take, offset;
		if (threadIdx.x == 0)
		{
			int ttake = singleThreadReserveRead(q, num);
			take = ttake;
			unsigned int toffset = ldg_cg(&front[q]);
			offset = toffset;
			stg_cg(&front[q], toffset + ttake);
		}
		__syncthreads();
		if (threadIdx.x < take)
			multiThreadRead(q, localElement, threadIdx.x, offset);
		return take;
	}

	__device__
	int dequeueWarp(int q, T* localElement, int num)
	{
		int take, offset, lid = laneid();
		if (lid == 0)
		{
			take = singleThreadReserveRead(q, num);
			offset = atomicAdd(&front[q], take);
		}
			
		take = __shfl_sync(~0U, take, 0);
		if (lid < take)
		{
			offset = __shfl_sync(~0U, offset, 0);
			multiThreadRead(q, localElement, threadIdx.x, offset);
		}
		return take;
	}

	__device__
	int dequeue(int q, T& element)
	{
		int readable = atomicSub(&count[q], 1);
		if (readable <= 0)
		{
			atomicAdd(&count[q], 1);
			return 0;
		}
		unsigned int pos = atomicAdd(&front[q], 1) % SIZE;

		T el;
		AccessControl:: template readread<SIZE, T>(el, indices[q] + pos, UNUSED);
		element = el;
		return 1;
	}

	__device__
	int size(int q)
	{
		return *const_cast<volatile int*>(&count[q]);
	}

	struct QueuePos
	{
		unsigned int pos;
	public:
		QueuePos() = default;
		
		__device__
		QueuePos(unsigned int pos) : pos(pos) {}

		__device__
		const QueuePos& operator += (unsigned int n)
		{
			pos = (pos + n) % SIZE;
			return *this;
		}

		__device__
		QueuePos operator + (unsigned int n)
		{
			return QueuePos((pos + n) % SIZE);
		}

		__device__
		bool operator == (const QueuePos& other)
		{
			return pos == other.pos;
		}

		__device__
		bool operator != (const QueuePos& other)
		{
			return pos != other.pos;
		}

		__device__
		T read(MultiIndexQueue& qs, int q)
		{
			T copy;
			while ((copy = ldg_cg(qs.indices[q] + pos)) == UNUSED)
				__threadfence();
			return copy;
		}

		__device__
		void write(MultiIndexQueue& qs, int q, const T& val)
		{
			stg_cg(qs.indices[q] + pos, val);
		}

		__device__
		int until(const QueuePos& other)
		{
			return (other.pos - pos + SIZE) % SIZE;
		}
	};
	
	__device__
	QueuePos begin(int q)
	{
		return QueuePos(ldg_cg(&front[q]));
	}

	__device__
	QueuePos end(int q)
	{
		return QueuePos(ldg_cg(&back[q]));
	}

	__device__
	void readState(int q, int &count, unsigned int &front, unsigned int &back)
	{
		count = this->count[q];
		front = this->front[q];
		back = this->back[q];
	}
};

//template <unsigned int SIZE, typename T = unsigned int, bool OVERFLOWCHECK = true, bool WAITFORFREE = false, bool AVOIDATOMICS = false>
//class IndexQueue
//{
//private:
//	static constexpr T UNUSED = static_cast<T>(-1);
//
//	T indices[SIZE];
//
//	int count;
//
//	unsigned int front;
//	unsigned int back;
//
//public:
//	__device__
//	void init()
//	{
//		if (blockIdx.x == 0 && threadIdx.x == 0)
//		{
//			count = 0;
//			front = back = 0U;
//		}
//
//		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SIZE; i += blockDim.x * gridDim.x)
//			indices[i] = UNUSED;
//	}
//
//	__device__
//	void enqueue(T i)
//	{
//		int fill = atomicAdd(&count, 1);
//		if (!OVERFLOWCHECK || fill < static_cast<int>(SIZE))
//		{
//			unsigned int pos = atomicInc(&back, SIZE - 1U);
//			if (OVERFLOWCHECK || WAITFORFREE)
//			{
//				if (AVOIDATOMICS)
//				{
//					while (ldg_cg(indices + pos) != UNUSED)
//						__threadfence();
//					stg_cg(indices + pos, i);
//					__threadfence();
//				}
//				else
//				{
//					while (atomicCAS(indices + pos, UNUSED, i) != UNUSED)
//						__threadfence();
//				}
//			}
//			else if (AVOIDATOMICS)
//			{
//				stg_cg(indices + pos, i);
//				__threadfence();
//			}
//			else
//				atomicExch(indices + pos, i);
//		}
//		else
//			__trap();
//	}
//
//	__device__
//	int singleThreadReserveRead(int& offset, int num)
//	{
//		int readable = atomicSub(&count, num);
//		if (readable < num)
//		{
//			int putback = min(num - readable, num);
//			atomicAdd(&count, putback);
//			num = num - putback;
//			// note: if could be removed -> trade useless atomicAdd vs if
//			if (num == 0)
//				return 0;
//		}
//		offset = atomicAdd(&front, num);
//		return num;
//	}
//
//	__device__
//	void multiThreadRead(T* localElement, int tid, int offset)
//	{
//		int pos = (offset + tid) % SIZE;
//		T el;
//		if (AVOIDATOMICS)
//		{
//			while ((el = ldg_cg(indices + pos)) == UNUSED)
//				__threadfence();
//			stg_cg(indices + pos, UNUSED);
//			__threadfence();
//		}
//		else
//		{
//			while ((el = atomicExch(indices + pos, UNUSED)) == UNUSED)
//				__threadfence();
//		}
//		*localElement = el;
//	}
//
//	__device__
//	int dequeueBlock(T* localElement, int num)
//	{
//		__shared__ int take, offset;
//		if (threadIdx.x == 0)
//			take = singleThreadReserveRead(offset, num);
//		__syncthreads();
//		if (threadIdx.x < take)
//			multiThreadRead(localElement, threadIdx.x, offset);
//		return take;
//	}
//
//	__device__
//	int dequeueWarp(T* localElement, int num)
//	{
//		int take, offset, lid = laneid();
//		if (lid == 0)
//			take = singleThreadReserveRead(offset, num);
//		take = __shfl_sync(~0U, take, 0);
//		if (lid < take)
//		{
//			offset = __shfl_sync(~0U, offset, 0);
//			multiThreadRead(localElement, threadIdx.x, offset);
//		}
//		return take;
//	}
//
//	__device__
//	int dequeue(T& element)
//	{
//		int readable = atomicSub(&count, 1);
//		if (readable <= 0)
//		{
//			atomicAdd(&count, 1);
//			return 0;
//		}
//		unsigned int pos = atomicAdd(&front, 1) % SIZE;
//		element = atomicExch(indices + pos, UNUSED);
//		return 1;
//	}
//
//	__device__ int size()
//	{
//		return *const_cast <volatile int*>(&count);
//	}
//
//	__device__ 
//	void readState(int &count, unsigned int &front, unsigned int &back)
//	{
//		count = this->count;
//		front = this->front;
//		back = this->back;
//	}
//};

#endif  // INCLUDED_CURE_INDEX_QUEUE
