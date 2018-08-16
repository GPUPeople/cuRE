


#ifndef INCLUDED_CURE_FRAMEBUFFER
#define INCLUDED_CURE_FRAMEBUFFER

#pragma once

#include <math/vector.h>

#include "config.h"

#ifndef FRAMEBUFFER_GLOBAL
#define FRAMEBUFFER_GLOBAL extern
#endif 

extern "C"
{
	FRAMEBUFFER_GLOBAL surface<void, cudaSurfaceType2D> color_buffer;
	FRAMEBUFFER_GLOBAL __constant__ uint2 color_buffer_size;

	//FRAMEBUFFER_GLOBAL __constant__ float* depth_buffer;
	FRAMEBUFFER_GLOBAL surface<void, cudaSurfaceType2D> depth_buffer;
	FRAMEBUFFER_GLOBAL __constant__ uint2 depth_buffer_size;
}

class FrameBuffer
{
	static __device__ unsigned char toLinear8(float c)
	{
		return static_cast<unsigned char>(saturate(c) * 255.0f);
	}

	static __device__ unsigned char toSRGB8(float c)
	{
		if (FRAMEBUFFER_SRGB)
			return toLinear8(powf(c, 1.0f / 2.2f));
		else
			return toLinear8(c);
	}

	static __device__ float fromLinear8(unsigned char c)
	{
		return c * (1.0f / 255.0f);
	}
	
	static __device__ float fromSRGB8(unsigned char c)
	{
		if (FRAMEBUFFER_SRGB)
			return powf(fromLinear8(c), 2.2f);
		else
			return fromLinear8(c);
	}

public:
	__device__
	static math::float4 readColor(int x, int y)
	{
		auto c = surf2Dread<uchar4>(color_buffer, 4 * x, y, cudaBoundaryModeZero);
		return { fromSRGB8(c.x), fromSRGB8(c.y), fromSRGB8(c.z), fromLinear8(c.w) };
	}

	__device__
	static void writeColor(cudaSurfaceObject_t surface, int x, int y, uchar4 c)
	{
		surf2Dwrite(c, surface, 4 * x, y, cudaBoundaryModeZero);
	}

	__device__
	static void writeColor(int x, int y, uchar4 c)
	{
		surf2Dwrite(c, color_buffer, 4 * x, y, cudaBoundaryModeZero);
	}
	
	__device__
	static void writeColor(cudaSurfaceObject_t surface, int x, int y, const math::float4& c)
	{
		writeColor(surface, x, y, uchar4 { toSRGB8(c.x), toSRGB8(c.y), toSRGB8(c.z), toLinear8(c.w) });
	}

	__device__
	static void writeColor(int x, int y, const math::float4& c)
	{
		writeColor(x, y, uchar4 { toSRGB8(c.x), toSRGB8(c.y), toSRGB8(c.z), toLinear8(c.w) });
	}

	template <class BlendOp>
	__device__
	static auto writeColor(int x, int y, const math::float4& src) -> decltype(BlendOp()(math::float4()), static_cast<void>(0))
	{
		if (BLENDING)
		{
			writeColor(x, y, BlendOp()(src));
			return;
		}

		writeColor(x, y, src);
	}

	template <class BlendOp>
	__device__
	static auto writeColor(int x, int y, const math::float4& src) -> decltype(BlendOp()(math::float4(), math::float4()), static_cast<void>(0))
	{
		if (BLENDING)
		{
			writeColor(x, y, BlendOp()(src, readColor(x, y)));
			return;
		}

		writeColor(x, y, src);
	}

	__device__
	static float readDepth(int x, int y)
	{
		return surf2Dread<float>(depth_buffer, 4 * x, y, cudaBoundaryModeZero);
		//return *(depth_buffer + depth_buffer_size.x * y + x);
	}

	__device__
	static void writeDepth(int x, int y, float depth)
	{
		surf2Dwrite(depth, depth_buffer, 4 * x, y, cudaBoundaryModeZero);
		//*(depth_buffer + depth_buffer_size.x * y + x) = depth;
	}
};

#endif  // INCLUDED_CURE_FRAMEBUFFER
