


#ifndef INCLUDED_PTX_PRIMITIVES
#define INCLUDED_PTX_PRIMITIVES

#pragma once

#include "ptx_primitives_64.cuh"


__device__ inline uint laneid()
{
	uint mylaneid;
	asm("mov.u32 %0, %laneid;" : "=r" (mylaneid));
	return mylaneid;
}

// requires ptx isa 1.3
__device__ inline uint warpid()
{
	uint mywarpid;
	asm("mov.u32 %0, %warpid;" : "=r" (mywarpid));
	return mywarpid;
}

__device__ inline uint nwarpid()
{
	uint mynwarpid;
	asm("mov.u32 %0, %nwarpid;" : "=r" (mynwarpid));
	return mynwarpid;
}

// requires ptx isa 1.3
__device__ inline uint smid()
{
	uint mysmid;
	asm("mov.u32 %0, %smid;" : "=r" (mysmid));
	return mysmid;
}

__device__ inline void volbar()
{
	asm volatile("membar.cta;");
}
	
#if __CUDA_ARCH__ < 300
__device__ inline uint gridid()
{
	uint mygridid;
	asm("mov.u32 %0, %gridid;" : "=r" (mygridid));
	return mygridid;
}
#else
__device__ inline unsigned long long gridid()
{
	unsigned long long mygridid;
	asm("mov.u64 %0, %gridid;" : "=l" (mygridid));
	return mygridid;
}
#endif

// requires ptx isa 2.0 and sm_20
__device__ inline uint nsmid()
{
	uint mynsmid;
	asm("mov.u32 %0, %nsmid;" : "=r" (mynsmid));
	return mynsmid;
}

// requires ptx isa 2.0 and sm_20
__device__ inline uint lanemask()
{
	uint lanemask;
	asm("mov.u32 %0, %lanemask_eq;" : "=r" (lanemask));
	return lanemask;
}

// requires ptx isa 2.0 and sm_20
__device__ inline uint lanemask_le()
{
	uint lanemask;
	asm("mov.u32 %0, %lanemask_le;" : "=r" (lanemask));
	return lanemask;
}

// requires ptx isa 2.0 and sm_20
__device__ inline uint lanemask_lt()
{
	uint lanemask;
	asm("mov.u32 %0, %lanemask_lt;" : "=r" (lanemask));
	return lanemask;
}

// requires ptx isa 2.0 and sm_20
__device__ inline uint lanemask_ge()
{
	uint lanemask;
	asm("mov.u32 %0, %lanemask_ge;" : "=r" (lanemask));
	return lanemask;
}

// requires ptx isa 2.0 and sm_20
__device__ inline uint lanemask_gt()
{
	uint lanemask;
	asm("mov.u32 %0, %lanemask_gt;" : "=r" (lanemask));
	return lanemask;
}

// custom sync
__device__ inline void syncthreads(uint lock = 0, int num = -1)
{
	if(num == -1)
	{
		asm volatile ("bar.sync %0;" : : "r"(lock));
	}
	else
	{
		asm volatile ("bar.sync %0, %1;" : : "r"(lock), "r"(num));
	}
}

__device__ inline void arrive(uint lock, uint num)
{
	asm volatile ("bar.arrive %0, %1;" : : "r"(lock), "r"(num));
}

__device__ inline uint syncthreads_count(uint predicate, uint lock = 0, int num = -1)
{
	uint res;
	if(num == -1)
	{
		asm volatile ("bar.red.popc.u32 %0, %1, %2;" : "=r" (res) : "r"(lock), "r" (predicate));
	}
	else
	{
		asm volatile ("bar.red.popc.u32 %0, %1, %2, %3;" : "=r" (res) : "r"(lock), "r"(num), "r" (predicate));
	}
	return res;
}

__device__ inline int syncthreads_or(int predicate, uint lock = 0, int num = -1)
{
	int res;
	if(num == -1)
	{
		asm volatile (
			"{                                 \n\t"
			"  .reg .pred p0;                  \n\t"
			"  .reg .pred p1;                  \n\t"
			"  setp.ne.s32 p0, 0, %2;          \n\t"
			"  bar.red.or.pred p1, %1, p0;     \n\t"
			"  selp.s32 %0, 1, 0, p1;          \n\t"
			"}" : "=r" (res) : "r"(lock), "r" (predicate));
	}
	else
	{
		asm volatile (
			"{                                 \n\t"
			"  .reg .pred p0;                  \n\t"
			"  .reg .pred p1;                  \n\t"
			"  setp.ne.s32 p0, 0, %3;          \n\t"
			"  bar.red.or.pred p1, %1, %2, p0; \n\t"
			"  selp.s32 %0, 1, 0, p1;          \n\t"
			"}" : "=r" (res) : "r"(lock), "r"(num), "r" (predicate));
	}
	return res;
}

__device__ inline int syncthreads_and(int predicate, uint lock = 0, int num = -1)
{
	int res;
	if(num == -1)
	{
		asm volatile ("bar.red.and.pred %0, %1, %2;" : "=r" (res) : "r"(lock), "r" (predicate));
	}
	else
	{
		asm volatile ("bar.red.and.pred %0, %1, %2, %3;" : "=r" (res) : "r"(lock), "r"(num), "r" (predicate));
	}
	return res;
}

__device__
inline unsigned long long ull_from_hilo(unsigned int hi, unsigned int lo)
{
	return __double_as_longlong(__hiloint2double(hi, lo));
}

__device__
inline unsigned int hi_from_ull(unsigned long long v)
{
	return __double2hiint(__longlong_as_double(v));
}

__device__
inline unsigned int lo_from_ull(unsigned long long v)
{
	return __double2loint(__longlong_as_double(v));
}

#endif  // INCLUDED_PTX_PRIMITIVES
