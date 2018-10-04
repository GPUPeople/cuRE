


#ifndef INCLUDED_CURE_SHADERS_SIMPLE
#define INCLUDED_CURE_SHADERS_SIMPLE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>
#include "../pipeline/viewport.cuh"
#include "../pipeline/fragment_shader_stage.cuh"
#include "noise.cuh"
#include "uniforms.cuh"


struct Vertex
{
	math::float4 position;
};

__device__
inline Vertex shfl(const Vertex& v, int src_lane)
{
	return { { __shfl_sync(~0U, v.position.x, src_lane), __shfl_sync(~0U, v.position.y, src_lane), __shfl_sync(~0U, v.position.z, src_lane), __shfl_sync(~0U, v.position.w, src_lane) } };
}

struct SimpleVertexShader
{
	__device__
	Vertex operator ()(const math::float3& v_p, const math::float3& v_n, math::float3& n, math::float3& l) const
	{
		//n = v_n;
		//auto v = v_p + sin(uniform[0]) * v_n * 40.1f;
		auto v = v_p;
		n = (camera.V * math::float4(v_n, 0)).xyz();
		//printf("vp: %f %f %f  vn %f %f %f\n", v_p.x, v_p.y, v_p.z, v_n.x, v_n.y, v_n.z);
		//n = math::float3(0, 1, 0);
		l = { 0,0,0 };
		return { camera.PVM * math::float4(v, 1.0f) };
	}
};


struct TexturedVertexShader
{
	__device__
	Vertex operator ()(const math::float3& v_p, const math::float3& v_n, const math::float2& v_t, math::float3& n, math::float2& t) const
	{
		//float offset = sin(uniform[0] * 3.14159265358979f) * 0.4f;
		//auto v = v_p + (offset * offset) * v_n;
		auto v = v_p;
		n = (camera.V * math::float4(v_n, 0)).xyz();
		t = v_t;
		return { camera.PVM * math::float4(v, 1.0f) };
	}
};


struct SimpleFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(const math::float3& n, const math::float3& l) const
	{
		//return math::float4(1.0f, 0.0f, 0.0f, 1.0f);
		//return math::float4(1.0f, 1.0f, 1.0f, 1.0f);
		return math::float4(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1.0f);

		//math::float3 ndx(dFdx(n.x), dFdx(n.y), dFdx(n.z));
		//math::float3 ndy(dFdy(n.x), dFdy(n.y), dFdy(n.z));
		//float lnx = length(ndx);
		//float lny = length(ndy);

		////return math::float4(10 * lnx, 10 * lny, 0.0f, 1.0f);
		//if (length(ndx)>0.00001f)
		//	return math::float4(abs(normalize(ndx)), 1.0f);
		//else
		//	return math::float4(0.0f, 0.0f, 0.0f, 1.0f);
		//return math::float4(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1.0f);

		//return math::float4(math::float3(1, 1, 1)*-n.z, 1.0f);
		//return math::float4(normalize(n), 1);
	}
};



struct VertexHeavyVertexShader
{
	__device__
	Vertex operator ()(const math::float3& v_p, const math::float3& v_n, math::float3& c) const
	{
		float noise = simplexNoiseFractal(v_p);

		c = { noise, noise, noise };

		return { camera.PVM * math::float4(v_p, 1.0f) };
	}
};

struct VertexHeavyFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(const math::float3& c) const
	{
		return { c, 1.0f };
	}
};

struct FragmentHeavyVertexShader
{
	__device__
	Vertex operator ()(const math::float3& v_p, const math::float3& v_n, math::float3& p) const
	{
		p = v_p;
		return { camera.PVM * math::float4(v_p, 1.0f) };
	}
};

struct FragmentHeavyFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(const math::float3& p) const
	{
		float noise = simplexNoiseFractal(p);

		return { noise, noise, noise, 1.0f };
	}
};


texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex;

template <bool WIREFRAME>
struct TexturedFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(const math::float3& n, const math::float2& t) const
	{
		//float4 c = tex2D(tex, t.x, t.y);
		auto du = make_float2(dFdx(t.x), dFdx(t.y));
		auto dv = make_float2(dFdy(t.x), dFdy(t.y));
		float4 c = tex2DGrad(tex, t.x, t.y, du, dv);
		float lambert = -n.z;
		math::float3 color = math::float3(c.x, c.y, c.z) * lambert;


		if (WIREFRAME)
		{
			float v = min(min(__f[0], __f[1]), __f[2]);

			if (v < 0.001f)
				return math::float4(0.0f, 0.0f, 0.0f, 1.0f);
		}

		return math::float4(color, 1.0f);
	}
};

#endif  // INCLUDED_CURE_SHADERS_SIMPLE
