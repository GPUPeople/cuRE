


#ifndef INCLUDED_PTX_PRIMITIVES_64
#define INCLUDED_PTX_PRIMITIVES_64

#pragma once

typedef unsigned int uint;


__device__ inline uint isshared(const void* p)
{
	uint res;
	asm("{\n\t"
	    ".reg .pred t1;\n\t"
	    "isspacep.shared t1, %1;\n\t"
	    "selp.u32 	%0, 1, 0, t1;\n\t"
	    "}\n\t"
	    : "=r"(res)
	    : "l"(p));
	return res;
}

__device__ inline uint islocal(const void* p)
{
	uint res;
	asm("{\n\t"
	    ".reg .pred t1;\n\t"
	    "isspacep.local t1, %1;\n\t"
	    "selp.u32 	%0, 1, 0, t1;\n\t"
	    "}\n\t"
	    : "=r"(res)
	    : "l"(p));
	return res;
}

__device__ inline uint isconst(const void* p)
{
	uint res;
	asm("{\n\t"
	    ".reg .pred t1;\n\t"
	    "isspacep.const t1, %1;\n\t"
	    "selp.u32 	%0, 1, 0, t1;\n\t"
	    "}\n\t"
	    : "=r"(res)
	    : "l"(p));
	return res;
}

__device__ inline uint isglobal(const void* p)
{
	uint res;
	asm("{\n\t"
	    ".reg .pred t1;\n\t"
	    "isspacep.global t1, %1;\n\t"
	    "selp.u32 	%0, 1, 0, t1;\n\t"
	    "}\n\t"
	    : "=r"(res)
	    : "l"(p));
	return res;
}


__device__
inline int ldg_ca(const int* src)
{
	int dest;
	asm("ld.global.ca.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long ldg_ca(const long* src)
{
	int dest;
	asm("ld.global.ca.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long long ldg_ca(const long long* src)
{
	long long dest;
	asm("ld.global.ca.s64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned int ldg_ca(const unsigned int* src)
{
	unsigned int dest;
	asm("ld.global.ca.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long ldg_ca(const unsigned long* src)
{
	unsigned long dest;
	asm("ld.global.ca.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long long ldg_ca(const unsigned long long* src)
{
	unsigned long long dest;
	asm("ld.global.ca.u64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline float ldg_ca(const float* src)
{
	float dest;
	asm("ld.global.ca.f32 %0, [%1];" : "=f"(dest) : "l"(src));
	return dest;
}


__device__
inline int ldg_cg(const int* src)
{
	int dest;
	asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long ldg_cg(const long* src)
{
	long dest;
	asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long long ldg_cg(const long long* src)
{
	long long dest;
	asm volatile("ld.global.cg.s64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned int ldg_cg(const unsigned int* src)
{
	unsigned int dest;
	asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long ldg_cg(const unsigned long* src)
{
	unsigned long dest;
	asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long long ldg_cg(const unsigned long long* src)
{
	unsigned long long dest;
	asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline float ldg_cg(const float* src)
{
	float dest;
	asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(dest) : "l"(src));
	return dest;
}


__device__
inline int ldg_cs(const int* src)
{
	int dest;
	asm("ld.global.cs.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long ldg_cs(const long* src)
{
	long dest;
	asm("ld.global.cs.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long long ldg_cs(const long long* src)
{
	long long dest;
	asm("ld.global.cs.s64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned int ldg_cs(const unsigned int* src)
{
	unsigned int dest;
	asm("ld.global.cs.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long ldg_cs(const unsigned long* src)
{
	unsigned long dest;
	asm("ld.global.cs.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long long ldg_cs(const unsigned long long* src)
{
	unsigned long long dest;
	asm("ld.global.cs.u64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline float ldg_cs(const float* src)
{
	float dest;
	asm("ld.global.cs.f32 %0, [%1];" : "=f"(dest) : "l"(src));
	return dest;
}

__device__
inline float2 ldg_cs(const float2* src)
{
	float2 dest;
	asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(dest.x), "=f"(dest.y) : "l"(src));
	return dest;
}

__device__
inline float4 ldg_cs(const float4* src)
{
	float4 dest;
	asm("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(dest.x), "=f"(dest.y), "=f"(dest.z), "=f"(dest.w) : "l"(src));
	return dest;
}

__device__
inline int ldg_lu(const int* src)
{
	int dest;
	asm("ld.global.lu.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long ldg_lu(const long* src)
{
	long dest;
	asm("ld.global.lu.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long long ldg_lu(const long long* src)
{
	long long dest;
	asm("ld.global.lu.s64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned int ldg_lu(const unsigned int* src)
{
	unsigned int dest;
	asm("ld.global.lu.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long ldg_lu(const unsigned long* src)
{
	unsigned long dest;
	asm("ld.global.lu.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long long ldg_lu(const unsigned long long* src)
{
	unsigned long long dest;
	asm("ld.global.lu.u64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline float ldg_lu(const float* src)
{
	float dest;
	asm("ld.global.lu.f32 %0, [%1];" : "=f"(dest) : "l"(src));
	return dest;
}


__device__
inline int ldg_cv(const int* src)
{
	int dest;
	asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long ldg_cv(const long* src)
{
	long dest;
	asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline long long ldg_cv(const long long* src)
{
	long long dest;
	asm volatile("ld.global.cv.s64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned int ldg_cv(const unsigned int* src)
{
	unsigned int dest;
	asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long ldg_cv(const unsigned long* src)
{
	unsigned long dest;
	asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(dest) : "l"(src));
	return dest;
}

__device__
inline unsigned long long ldg_cv(const unsigned long long* src)
{
	unsigned long long dest;
	asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(dest) : "l"(src));
	return dest;
}

__device__
inline float ldg_cv(const float* src)
{
	float dest;
	asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(dest) : "l"(src));
	return dest;
}



__device__
inline const int& stg_wb(int* dest, const int& src)
{
	asm volatile("st.global.wb.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long& stg_wb(long* dest, const long& src)
{
	asm volatile("st.global.wb.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long long& stg_wb(long long* dest, const long long& src)
{
	asm volatile("st.global.wb.s64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const unsigned int& stg_wb(unsigned int* dest, const unsigned int& src)
{
	asm volatile("st.global.wb.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long& stg_wb(unsigned long* dest, const unsigned long& src)
{
	asm volatile("st.global.wb.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long long& stg_wb(unsigned long long* dest, const unsigned long long& src)
{
	asm volatile("st.global.wb.u64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const float& stg_wb(float* dest, const float& src)
{
	asm volatile("st.global.wb.f32 [%0], %1;" : : "l"(dest), "f"(src));
	return src;
}


__device__
inline const int& stg_cg(int* dest, const int& src)
{
	asm volatile("st.global.cg.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long& stg_cg(long* dest, const long& src)
{
	asm volatile("st.global.cg.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long long& stg_cg(long long* dest, const long long& src)
{
	asm volatile("st.global.cg.s64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const unsigned int& stg_cg(unsigned int* dest, const unsigned int& src)
{
	asm volatile("st.global.cg.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long& stg_cg(unsigned long* dest, const unsigned long& src)
{
	asm volatile("st.global.cg.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long long& stg_cg(unsigned long long* dest, const unsigned long long& src)
{
	asm volatile("st.global.cg.u64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const float& stg_cg(float* dest, const float& src)
{
	asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(dest), "f"(src));
	return src;
}


__device__
inline const int& stg_cs(int* dest, const int& src)
{
	asm("st.global.cs.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long& stg_cs(long* dest, const long& src)
{
	asm("st.global.cs.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long long& stg_cs(long long* dest, const long long& src)
{
	asm("st.global.cs.s64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const unsigned int& stg_cs(unsigned int* dest, const unsigned int& src)
{
	asm("st.global.cs.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long& stg_cs(unsigned long* dest, const unsigned long& src)
{
	asm("st.global.cs.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long long& stg_cs(unsigned long long* dest, const unsigned long long& src)
{
	asm("st.global.cs.u64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const float& stg_cs(float* dest, const float& src)
{
	asm("st.global.cs.f32 [%0], %1;" : : "l"(dest), "f"(src));
	return src;
}


__device__
inline const int& stg_wt(int* dest, const int& src)
{
	asm volatile("st.global.wt.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long& stg_wt(long* dest, const long& src)
{
	asm volatile("st.global.wt.s32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const long long& stg_wt(long long* dest, const long long& src)
{
	asm volatile("st.global.wt.s64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const unsigned int& stg_wt(unsigned int* dest, const unsigned int& src)
{
	asm volatile("st.global.wt.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long& stg_wt(unsigned long* dest, const unsigned long& src)
{
	asm volatile("st.global.wt.u32 [%0], %1;" : : "l"(dest), "r"(src));
	return src;
}

__device__
inline const unsigned long long& stg_wt(unsigned long long* dest, const unsigned long long& src)
{
	asm volatile("st.global.wt.u64 [%0], %1;" : : "l"(dest), "l"(src));
	return src;
}

__device__
inline const float& stg_wt(float* dest, const float& src)
{
	asm volatile("st.global.wt.f32 [%0], %1;" : : "l"(dest), "f"(src));
	return src;
}

#endif // INCLUDED_PTX_PRIMITIVES_64
