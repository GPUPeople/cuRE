


#ifndef INCLUDED_CURE_TRIANGLE_BUFFER
#define INCLUDED_CURE_TRIANGLE_BUFFER

#pragma once

#include <meta_utils.h>
#include <utils.h>

#include <math/vector.h>
#include <math/matrix.h>

#include <ptx_primitives.cuh>


template <unsigned int SIZE, unsigned int NUM_INTERPOLATORS, bool CHECKOVERFLOW = true, bool TRACK_FILL_LEVEL = false>
class TriangleBuffer
{
private:
	static constexpr unsigned int ELEMENT_SIZE = static_divup<16 * 4U + static_divup<NUM_INTERPOLATORS, 4>::value * 12 * 4U, 16U>::value * 16U;

	unsigned int next;

	int fill_level;
	int max_fill_level;

	unsigned int reference_counter[SIZE];
	__align__(16U) char buffer[ELEMENT_SIZE * SIZE];  // here be compiler bugs: screwed up address computations for RED.E.ADD on Maxwell when buffer goes before reference_counter


	__device__
	const char* data(unsigned int i) const
	{
		return buffer + i * ELEMENT_SIZE;
	}

	__device__
	char* data(unsigned int i)
	{
		return buffer + i * ELEMENT_SIZE;
	}

public:
	__device__
	void init()
	{
		if (blockIdx.x == 0 && threadIdx.x == 0)
		{
			next = 0U;

			if (TRACK_FILL_LEVEL)
			{
				fill_level = 0;
				max_fill_level = 0;
			}
		}

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SIZE; i += gridDim.x * blockDim.x)
			reference_counter[i] = 0U;
	}

	__device__
	void writeMaxFillLevel(int* dest)
	{
		if (blockIdx.x == 0 && threadIdx.x == 0)
			*dest = max_fill_level;
	}


	__device__
	void acquire(unsigned int i, unsigned int c = 1U)
	{
		if (CHECKOVERFLOW)
			while (atomicCAS(&reference_counter[i], 0U, c) != 0U)
				__threadfence();
		__threadfence();
	}

	__device__
	unsigned int allocate(unsigned int c = 1U)
	{
		if (TRACK_FILL_LEVEL)
			atomicMax(&max_fill_level, atomicAdd(&fill_level, 1));

		unsigned int i = atomicInc(&next, SIZE - 1);
		acquire(i, c);

		return i;
	}

	__device__
	unsigned int allocateWarp(unsigned int c = 1U)
	{
		unsigned int mask = __ballot_sync(~0U, 1);

		unsigned int before_me = mask & lanemask_lt();
		int num = __popc(mask);
		int lead = __ffs(mask) - 1;
		int me = __popc(before_me);

		unsigned int i;
		if (me == 0U)
		{
			if (TRACK_FILL_LEVEL)
				atomicMax(&max_fill_level, atomicAdd(&fill_level, num));

			while ((i = atomicAdd(&next, num) % SIZE) + num > SIZE);
		}

		i = __shfl_sync(~0U, i, lead) + me;

		acquire(i, c);
		return i;
	}

	__device__
	unsigned int addReference(unsigned int i, unsigned int c = 1U)
	{
		if (CHECKOVERFLOW)
			return atomicAdd(&reference_counter[i], c) + c;
		return c;
	}

	__device__
	unsigned int release(unsigned int i, unsigned int c = 1U)
	{
		__threadfence();
		if (CHECKOVERFLOW)
		{
			unsigned int refcount = atomicSub(&reference_counter[i], c) - c;

			if (TRACK_FILL_LEVEL)
				if (refcount == 0U)
					atomicSub(&fill_level, 1);

			return refcount;
		}
		else
			if (TRACK_FILL_LEVEL)
				atomicSub(&fill_level, 1);

		return 0;
	}


	__device__
	void storeTriangle(unsigned int i, const math::float3x3& M, const math::float3& uz, const math::int4& bounds)
	{
		float4* d = reinterpret_cast<float4*>(data(i));
		d[0] = make_float4(M._11, M._12, M._13, uz.x);
		d[1] = make_float4(M._21, M._22, M._23, uz.y);
		d[2] = make_float4(M._31, M._32, M._33, uz.z);
		d[3] = make_float4(__int_as_float(bounds.x), __int_as_float(bounds.y), __int_as_float(bounds.z), __int_as_float(bounds.w));
	}

	__device__
	void storeInterpolator(unsigned int j, unsigned int i, const math::float4x3& M)
	{
		float4* d = reinterpret_cast<float4*>(data(i)) + 4;

		//d[3 * j + 0] = make_float4(M._11, M._21, M._31, M._41);
		//d[3 * j + 1] = make_float4(M._12, M._22, M._32, M._42);
		//d[3 * j + 2] = make_float4(M._13, M._23, M._33, M._43);
		d[3 * j + 0] = make_float4(M._11, M._12, M._13, M._21);
		d[3 * j + 1] = make_float4(M._22, M._23, M._31, M._32);
		d[3 * j + 2] = make_float4(M._33, M._41, M._42, M._43);
	}

	__device__
	void loadTriangle(unsigned int i, math::float3x3* M, math::float3* uz, math::int4* bounds) const
	{
		const float4* d = reinterpret_cast<const float4*>(data(i));

		float4 d0 = d[0];
		float4 d1 = d[1];
		float4 d2 = d[2];
		float4 d3 = d[3];

		M->_11 = d0.x; M->_12 = d0.y; M->_13 = d0.z;
		M->_21 = d1.x; M->_22 = d1.y; M->_23 = d1.z;
		M->_31 = d2.x; M->_32 = d2.y; M->_33 = d2.z; 

		uz->x = d0.w;
		uz->y = d1.w;
		uz->z = d2.w;

		bounds->x = __float_as_int(d3.x);
		bounds->y = __float_as_int(d3.y);
		bounds->z = __float_as_int(d3.z);
		bounds->w = __float_as_int(d3.w);
	}

	__device__
	math::int4 loadBounds(unsigned int i) const
	{
		const float4* d = reinterpret_cast<const float4*>(data(i));
		float4 d3 = d[3];
		return math::float4(__float_as_int(d3.x), __float_as_int(d3.y), __float_as_int(d3.z), __float_as_int(d3.w));
	}

	__device__
	math::float3x3 loadM(unsigned int i) const
	{
		const float4* d = reinterpret_cast<const float4*>(data(i));

		float4 d0 = d[0];
		float4 d1 = d[1];
		float4 d2 = d[2];

		math::float3x3 M;

		M._11 = d0.x; M._12 = d0.y; M._13 = d0.z;
		M._21 = d1.x; M._22 = d1.y; M._23 = d1.z;
		M._31 = d2.x; M._32 = d2.y; M._33 = d2.z;

		return M;
	}

	__device__
	math::float3 loadEdge(unsigned int i, int edge) const
	{
		const float4* d = reinterpret_cast<const float4*>(data(i));
		float4 e = d[edge];

		return math::float3(e.x, e.y, e.z);
	}

	__device__
	void loadInterpolator(math::float4x3* M, unsigned int N, unsigned int i) const
	{
		const float4* d = reinterpret_cast<const float4*>(data(i)) + 4;

		float4* mlocal = reinterpret_cast<float4*>(M);
		for (int j = 0; j < N; ++j)
		{
			mlocal[3 * j + 0] = d[3 * j + 0];
			mlocal[3 * j + 1] = d[3 * j + 1];
			mlocal[3 * j + 2] = d[3 * j + 2];
		}
	}

	__device__
	void loadInterpolatorsWarp(math::float4x3* M, unsigned int N, unsigned int i) const
	{
		static_assert(sizeof(math::float4x3) == 48, "ERROR: assuming float4x3 is tightly packed");

		const float4* d = reinterpret_cast<const float4*>(data(i)) + 4;

		unsigned int num_floats = 12 * N;

		for (int j = 0; j < num_floats; j += WARP_SIZE)
		{
			unsigned int k = j + laneid();
			if (k < num_floats)
				(&M->_11)[k] = (&d->x)[k];
		}
	}

	//__device__
	//void loadTriangleWarp(unsigned int i, math::float3x3* M, math::float3* uz, math::int4* bounds) const
	//{
	//	int lid = laneid();
	//	if (lid == 0)
	//	{
	//		const float4* d = reinterpret_cast<const float4*>(data(i));
	//		//printf("%p\n", d);
	//		//__threadfence();
	//		M->_11 = d[0].x; M->_12 = d[0].y; M->_13 = d[0].z; uz->x = d[0].w;
	//		M->_21 = d[1].x; M->_22 = d[1].y; M->_23 = d[1].z; uz->y = d[1].w;
	//		M->_31 = d[2].x; M->_32 = d[2].y; M->_33 = d[2].z; uz->z = d[2].w;
	//		bounds->x = __float_as_int(d[3].x); bounds->y = __float_as_int(d[3].y); bounds->z = __float_as_int(d[3].z); bounds->w = __float_as_int(d[3].w);
	//	}
	//}

	__device__
	void loadTriangleWarp(unsigned int i, math::float3x3* M, math::float3* uz, math::int4* bounds) const
	{
		int lid = laneid();
		if (lid < 16)
		{
			float d = *(reinterpret_cast<const float*>(data(i)) + lid);
			int j = lid / 4;
			int k = lid % 4;

			if (j < 3)
			{
				float* m = (k < 3) ? &M->_11 + 3 * j + k : &uz->x + j;
				*m = d;
			}
			else
			{
				(&bounds->x)[k] = __float_as_int(d);
			}
		}
	}

	__device__
	void loadTriangleWarp(unsigned int i, math::float3x3* M, math::float3* uz) const
	{
		int lid = laneid();
		if (lid < 12)
		{
			float d = *(reinterpret_cast<const float*>(data(i)) + lid);
			int j = lid / 4;
			int k = lid % 4;

			float* m = (k < 3) ? &M->_11 + 3 * j + k : &uz->x + j;
			*m = d;
		}
	}


	__device__
	void check()
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SIZE; i += gridDim.x * blockDim.x)
		{
			if (reference_counter[i] != 0U)
				printf("Triangle Buffer corruppted at: %d : %d\n", i, reference_counter[i]);
		}
	}

};

#endif  // INCLUDED_CURE_TRIANGLE_BUFFER
