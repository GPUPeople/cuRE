


#ifndef INCLUDED_CURE_VIEWPORT
#define INCLUDED_CURE_VIEWPORT

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "config.h"


struct Viewport
{
	float left;
	float top;
	float right;
	float bottom;
};

extern "C"
{
	extern __constant__ Viewport viewport;

	extern __constant__ float4 pixel_scale;
}

__device__
static math::int4 computeRasterBounds(const math::float2& bounds_min, const math::float2& bounds_max)
{
	float vp_scale_x = 0.5f * (viewport.right - viewport.left);
	float vp_scale_y = 0.5f * (viewport.bottom - viewport.top);

	float x_min = max((bounds_min.x + 1.0f) * vp_scale_x + viewport.left, viewport.left);
	float y_min = max((bounds_min.y + 1.0f) * vp_scale_y + viewport.top, viewport.top);

	float x_max = min((bounds_max.x + 1.0f) * vp_scale_x + viewport.left, viewport.right);
	float y_max = min((bounds_max.y + 1.0f) * vp_scale_y + viewport.top, viewport.bottom);

	return math::int4(ceil(x_min - 0.5f), ceil(y_min - 0.5f), floor(x_max + 0.5f), floor(y_max + 0.5f));
}

__device__
static math::float3 clipcoordsFromRaster(int x, int y)
{
	return math::float3(x * pixel_scale.x + pixel_scale.z, y * pixel_scale.y + pixel_scale.w, 1.0f);
}

__device__
static math::float2 rastercoordsFromClip(float x, float y)
{
	float vp_scale_x = 0.5f * (viewport.right - viewport.left);
	float vp_scale_y = 0.5f * (viewport.bottom - viewport.top);

	return math::float2((x + 1.0f)* vp_scale_x + viewport.left, (y + 1.0f)* vp_scale_y + viewport.top);
}


struct RasterToClipConverter
{
	__device__
	static math::float2 point(const math::float2& p)
	{
		return math::float2(p.x * pixel_scale.x + pixel_scale.z, p.y * pixel_scale.y + pixel_scale.w);
	}
	__device__
	static math::float2 vec(const math::float2& v)
	{
		return math::float2(v.x * pixel_scale.x, v.y * pixel_scale.y);
	}
};


#endif  // INCLUDED_CURE_VIEWPORT
