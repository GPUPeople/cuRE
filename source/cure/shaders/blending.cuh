


#ifndef INCLUDED_CURE_SHADERS_BLENDING
#define INCLUDED_CURE_SHADERS_BLENDING

#pragma once

#include <math/vector.h>
#include <math/matrix.h>
#include "../pipeline/viewport.cuh"
#include "../pipeline/fragment_shader_stage.cuh"
#include "uniforms.cuh"


struct BlendVertex
{
	math::float4 position;
};

__device__
inline BlendVertex shfl(const BlendVertex& v, int src_lane)
{
	return { { __shfl_sync(~0U, v.position.x, src_lane), __shfl_sync(~0U, v.position.y, src_lane), __shfl_sync(~0U, v.position.z, src_lane), __shfl_sync(~0U, v.position.w, src_lane) } };
}


struct BlendVertexShader
{
	__device__
	BlendVertex operator ()(const math::float2& p, const math::float3& n, const math::float3& c, math::float3& color) const
	{
		color = c;
		return { math::float4(p.x, p.y, 0.01f, 1) };
	}
};

struct IsoBlendVertexShader
{
	__device__
	BlendVertex operator ()(const math::float3& v_p, const math::float3& v_n, const math::float4& v_c, math::float3& n, math::float4& c) const
	{
		n = (camera.V * math::float4(v_n, 0)).xyz();

		c = v_c;

		return{ camera.PVM * math::float4(v_p, 1.0f) };
	}
};

struct GlyphVertexShader
{
	__device__
	BlendVertex operator ()(const math::float3& v_p, const math::float3& v_uvsign, const math::float4& v_c, math::float3& uvsign, math::float4& c) const
	{
		uvsign = v_uvsign;

		c = v_c;

		return{ camera.PVM * math::float4(v_p, 1.0f) };
	}
};


struct BlendFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& color)
	{
		return math::float4(color, 1.0f);
	}
};

struct IsoBlendFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& normal, const math::float4& color)
	{
		return math::float4(color * -normalize(normal).z);
	}
};

struct GlyphFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator()(const math::float3& uvsign, const math::float4& color)
	{
		if (uvsign.z < -5)
		{
			if ((uvsign.x * uvsign.x + uvsign.y * uvsign.y) >= 1.f)
				discard();
		}
		else
		{
			if ((uvsign.x*uvsign.x - uvsign.y) * uvsign.z > 0.0f)
				discard();
		}

		return math::float4(color.xyz(), 1.0f);
	}
};

#endif  // INCLUDED_CURE_SHADERS_BLENDING
