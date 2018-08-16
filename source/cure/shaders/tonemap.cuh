


#ifndef INCLUDED_TONEMAP
#define INCLUDED_TONEMAP

#pragma once

#include <math/vector.h>

constexpr float A = 0.15f;
constexpr float B = 0.50f;
constexpr float C = 0.10f;
constexpr float D = 0.20f;
constexpr float E = 0.02f;
constexpr float F = 0.30f;
//constexpr float W = 1.2f;
constexpr float W = 0.4f;
constexpr float exposure = 0.25f;


inline __device__ math::float3 Uncharted2Tonemap(const math::float3& x)
{
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

inline __device__ math::float3 tonemap(const math::float3& texColor)
{
	math::float3 curr = Uncharted2Tonemap(exposure * texColor);

	math::float3 whiteScale = rcp(Uncharted2Tonemap(math::float3(W, W, W)));
	math::float3 color = curr * whiteScale;

	return color;
}

#endif  // INCLUDED_TONEMAP
