


#ifndef INCLUDED_CURE_SHADERS_EYECANDY
#define INCLUDED_CURE_SHADERS_EYECANDY

#pragma once

#include <math/vector.h>
#include <math/matrix.h>
#include "../pipeline/viewport.cuh"
#include "../pipeline/fragment_shader_stage.cuh"
#include "tonemap.cuh"
#include "noise.cuh"
#include "uniforms.cuh"


struct EyeCandyVertexShader
{
	__device__
	Vertex operator ()(const math::float4& v, const math::float4& normal, const math::float4& color, math::float3& p, math::float3& n, math::float3& c) const
	{
		p = v.xyw();
		n = normal.xyz();
		c = color.xyz();
		return { v };
	}
};


struct EyeCandyFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& pos, const math::float3& normal, const math::float3& color)
	{
		//if (color.x == 0.0f && color.y == 0.0f && color.z == 0.0f)
		//	discard();

		math::float3 n = normalize(normal);

		math::float4 res = math::float4(color*(fabsf(n.z) + 0.2f), 1.0f);

		if (WIREFRAME)
		{
			float v = min(min(__f[0], __f[1]), __f[2]);
			if (v < 0.001f)
			{
				res *= 1.6f;
			}
		}

		return res;
	}
};

struct EyeCandyCoverageFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& pos, const math::float3& normal, const math::float3& color)
	{
		//if (color.x == 0.0f && color.y == 0.0f && color.z == 0.0f)
		//	discard();

		math::uint2 tile_loc = 8 * (__pixel_loc / 8) + 4;

		math::float2 c{ tile_loc.x - (viewport.left + viewport.right) * 0.5f, tile_loc.y - (viewport.top + viewport.bottom) * 0.5f };

		if ((dot(c, c) > CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS) && ((__pixel_loc.x % 2) ^ (__pixel_loc.y % 2)))
		{
			discard();
		}

		math::float4 res = math::float4(color*(-normal.z + 0.2f), 1.0f);

		float v = min(min(__f[0], __f[1]), __f[2]);
		if (v < 0.001f)
		{
			res *= 1.6f;
		}

		return res;
	}
};

struct EyeCandyQuadCoverageFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& pos, const math::float3& normal, const math::float3& color)
	{
		//if (color.x == 0.0f && color.y == 0.0f && color.z == 0.0f)
		//	discard();

		math::uint2 tile_loc = 8 * (__pixel_loc / 8) + 4;

		math::float2 c{ tile_loc.x - (viewport.left + viewport.right) * 0.5f, tile_loc.y - (viewport.top + viewport.bottom) * 0.5f };

		if ((dot(c, c) > CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS) && (((__pixel_loc.x / 2U) % 2) ^ ((__pixel_loc.y / 2U) % 2)))
		{
			discard();
		}

		math::float4 res = math::float4(color*(-normal.z + 0.2f), 1.0f);

		float v = min(min(__f[0], __f[1]), __f[2]);
		if (v < 0.001f)
		{
			res *= 1.6f;
		}

		return res;
	}
};


struct EyeCandyVertexShaderVertexHeavy
{
	__device__
	Vertex operator ()(const math::float4& v, const math::float4& normal, const math::float4& color, math::float3& c) const
	{
		float noise = simplexNoiseFractal(v.xyw());

		c = { noise, noise, noise };

		return { v };
	}
};

struct EyeCandyFragmentShaderVertexHeavy : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& color) //const
	{
		return { color, 1.0f };
	}
};

struct EyeCandyFragmentShaderFragmentHeavy : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& pos) //const
	{
		float noise = simplexNoiseFractal(pos);

		return { noise, noise, noise, 1.0f };
	}
};

#endif  // INCLUDED_CURE_SHADERS_EYECANDY
