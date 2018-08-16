


#ifndef INCLUDED_CURE_CLIPPING
#define INCLUDED_CURE_CLIPPING

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include <ptx_primitives.cuh>

#include "config.h"
#include "common.cuh"

#include "instrumentation.cuh"

#include "geometry_stage.cuh"
#include "vertex_shader_stage.cuh"
#include "rasterization_stage.cuh"


__device__
inline math::float3 hom_min(const math::float3& a, const math::float3& b)
{
	return math::float3(min(a.x * b.z, b.x * a.z), min(a.y * b.z, b.y * a.z), a.z * b.z);
}

__device__
inline math::float3 hom_max(const math::float3& a, const math::float3& b)
{
	return math::float3(max(a.x * b.z, b.x * a.z), max(a.y * b.z, b.y * a.z), a.z * b.z);
}

__device__
inline math::float2 proj(const math::float3& v)
{
	return (1.0f / v.z) * math::float2(v.x, v.y);
}


__device__
inline math::float3 clipEdgeNear(const math::float4& p1, const math::float4& p2)
{
	float t = (-p1.w - p1.z) / (p2.z - p1.z + p2.w - p1.w);
	return math::float3(math::lerp(p1.x, p2.x, t), math::lerp(p1.y, p2.y, t), math::lerp(p1.w, p2.w, t));
}

__device__
inline math::float3 clipEdgeFar(const math::float4& p1, const math::float4& p2)
{
	float t = (p1.w - p1.z) / (p2.z - p1.z + p1.w - p2.w);
	return math::float3(math::lerp(p1.x, p2.x, t), math::lerp(p1.y, p2.y, t), math::lerp(p1.w, p2.w, t));
}

__device__
inline math::float3 clipEdge(const math::float4& p1, const math::float4& p2, unsigned int outcode)
{
#if 0
	float a = outcode == 1U ? (-p1.w - p1.z) : (p1.w - p1.z);
	float b = outcode == 1U ? (p2.z - p1.z + p2.w - p1.w) : (p2.z - p1.z + p1.w - p2.w);
	float t = a / b;
	return math::float3(math::lerp(p1.x, p2.x, t), math::lerp(p1.y, p2.y, t), math::lerp(p1.w, p2.w, t));
#else
	if (outcode == 1U)
		return clipEdgeNear(p1, p2);

	return clipEdgeFar(p1, p2);
#endif
}

__device__
inline unsigned int computeOutcode(const math::float4& p)
{
#if 0
	unsigned int near_code = __float_as_uint(p.z + p.w) >> 31U;
	unsigned int far_code = (__float_as_uint(p.w - p.z) >> 30U) & 0x2U;
#else
	unsigned int near_code = p.z < -p.w ? 1U : 0U;
	unsigned int far_code = p.z > p.w ? 2U : 0U;
#endif
	return near_code | far_code;
}


__device__
inline void clipCorner(const math::float4& p1, const math::float4& p2, const math::float4& p3, unsigned int outcode_1, unsigned int outcode_2, unsigned int outcode_3, math::float3& p_11, math::float3& p_12)
{
	if (outcode_1 == 0U)
	{
		p_11 = p_12 = p1.xyw();
		return;
	}

	if ((outcode_1 ^ outcode_2) != 0)
		p_11 = clipEdge(p1, p2, outcode_1);

	if ((outcode_1 ^ outcode_3) != 0)
		p_12 = clipEdge(p1, p3, outcode_1);

	if ((outcode_1 ^ outcode_2) == 0)
		p_11 = p_12;

	if ((outcode_1 ^ outcode_3) == 0)
		p_12 = p_11;
}

__device__
bool clipTriangle(const math::float4& p1, const math::float4& p2, const math::float4& p3, math::float2& bounds_min, math::float2& bounds_max)
{
	if (!CLIPPING)
	{
		bounds_min = proj(hom_min(p1.xyw(), hom_min(p2.xyw(), p3.xyw())));
		bounds_max = proj(hom_max(p1.xyw(), hom_max(p2.xyw(), p3.xyw())));
		return false;
	}

	unsigned int outcode_1 = computeOutcode(p1);
	unsigned int outcode_2 = computeOutcode(p2);
	unsigned int outcode_3 = computeOutcode(p3);

	if ((outcode_1 ^ outcode_2) == 0U && (outcode_1 ^ outcode_3) == 0U)
	{
		bounds_min = proj(hom_min(p1.xyw(), hom_min(p2.xyw(), p3.xyw())));
		bounds_max = proj(hom_max(p1.xyw(), hom_max(p2.xyw(), p3.xyw())));
		return outcode_1 != 0U;
	}

	math::float3 p_11, p_12;
	clipCorner(p1, p2, p3, outcode_1, outcode_2, outcode_3, p_11, p_12);
	math::float3 p_21, p_22;
	clipCorner(p2, p3, p1, outcode_2, outcode_3, outcode_1, p_21, p_22);
	math::float3 p_31, p_32;
	clipCorner(p3, p1, p2, outcode_3, outcode_1, outcode_2, p_31, p_32);

	bounds_min = proj(hom_min(p_11, hom_min(p_12, hom_min(p_21, hom_min(p_22, hom_min(p_31, p_32))))));
	bounds_max = proj(hom_max(p_11, hom_max(p_12, hom_max(p_21, hom_max(p_22, hom_max(p_31, p_32))))));

	return false;
}

#endif  // INCLUDED_CURE_CLIPPING
