


#ifndef INCLUDED_CURE_SHADERS_CHECKERBOARD
#define INCLUDED_CURE_SHADERS_CHECKERBOARD

#pragma once

#include <math/vector.h>
#include <math/matrix.h>
#include "../pipeline/viewport.cuh"
#include "../pipeline/fragment_shader_stage.cuh"
#include "uniforms.cuh"


struct CheckerboardQuadCoverageShader : CoverageShader
{
	using CoverageShader::CoverageShader;

	template <typename Bitmask>
	__device__
	Bitmask operator()(Bitmask mask) const
	{
		math::float2 c = { (__tile_bounds.x + __tile_bounds.z - viewport.left - viewport.right) * 0.5f, (__tile_bounds.y + __tile_bounds.w - viewport.top - viewport.bottom) * 0.5f };

		if (dot(c, c) > CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS)
		{
			mask &= 0xCCCC3333CCCC3333;
		}
		return mask;
	}
};

struct CheckerboardCoverageShader : CoverageShader
{
	using CoverageShader::CoverageShader;

	template <typename Bitmask>
	__device__
	Bitmask operator()(Bitmask mask) const
	{
		math::float2 c = { (__tile_bounds.x + __tile_bounds.z - viewport.left - viewport.right) * 0.5f, (__tile_bounds.y + __tile_bounds.w - viewport.top - viewport.bottom) * 0.5f };

		if (dot(c, c) > CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS)
		{
			mask &= 0b1010101001010101101010100101010110101010010101011010101001010101;
		}
		return mask;
	}
};

#endif  // INCLUDED_CURE_FRAGMENT_SHADERS
