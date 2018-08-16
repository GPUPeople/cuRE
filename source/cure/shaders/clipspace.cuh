


#ifndef INCLUDED_CURE_SHADERS_CLIPSPACE
#define INCLUDED_CURE_SHADERS_CLIPSPACE

#pragma once

#include <math/vector.h>
#include "../pipeline/fragment_shader_stage.cuh"
#include "noise.cuh"


struct ClipspaceVertexShader
{
	__device__
	Vertex operator ()(const math::float4& v, math::float4& p) const
	{
		p = v;
		return { v };
	}
};

struct ClipspaceFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(const math::float4& p) const
	{
		float c = p.w * 0.01f;
		return math::float4(c, c, c, 1.0f);
	}
};


struct VertexHeavyClipspaceVertexShader
{
	__device__
	Vertex operator ()(const math::float4& v, math::float3& c) const
	{
		float noise = simplexNoiseFractal(v.xyw());

		c = { noise, noise, noise };

		return { v };
	}
};


struct VertexHeavyClipspaceFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(const math::float3& c) const
	{
		return math::float4(c, 1.0f);
	}
};

struct FragmentHeavyClipspaceFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(const math::float4& p) const
	{
		float noise = simplexNoiseFractal(p.xyw());

		return math::float4(noise, noise, noise, 1.0f);
	}
};

#endif  // INCLUDED_CURE_SHADERS_CLIPSPACE
