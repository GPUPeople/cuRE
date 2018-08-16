


#ifndef INCLUDED_CURE_SHADERS_OCEAN
#define INCLUDED_CURE_SHADERS_OCEAN

#pragma once

#include <math/vector.h>
#include <math/matrix.h>
#include "../pipeline/viewport.cuh"
#include "../pipeline/fragment_shader_stage.cuh"
#include "tonemap.cuh"
#include "noise.cuh"
#include "uniforms.cuh"


struct WaveVertex
{
	math::float4 position;
	math::float3 p;
	math::float3 n;
	//float h;
};

__device__
inline WaveVertex shfl(const WaveVertex& v, int src_lane)
{
	//return{ { __shfl_sync(~0U, v.position.x, src_lane), __shfl_sync(~0U, v.position.y, src_lane), __shfl_sync(~0U, v.position.z, src_lane), __shfl_sync(~0U, v.position.w, src_lane) }, __shfl_sync(~0U, v.h, src_lane) };
	return { { __shfl_sync(~0U, v.position.x, src_lane), __shfl_sync(~0U, v.position.y, src_lane), __shfl_sync(~0U, v.position.z, src_lane), __shfl_sync(~0U, v.position.w, src_lane) },{ __shfl_sync(~0U, v.p.x, src_lane), __shfl_sync(~0U, v.p.y, src_lane), __shfl_sync(~0U, v.p.z, src_lane) },{ __shfl_sync(~0U, v.n.x, src_lane), __shfl_sync(~0U, v.n.y, src_lane), __shfl_sync(~0U, v.n.z, src_lane) } };
}

struct WaterVertexShader
{
	__device__
	float readWave(uint32_t i, float& Q, float& A, math::float2& D, float& w, float& v, float mult_A, const math::float2& pos, float t) const
	{
		float boop = 0.0f;

		uint32_t off = 4 + i * 6;
		float phi;
		Q = uniform[off + 0];

		A = mult_A * uniform[off + 1];
		w = uniform[off + 2];
		phi = uniform[off + 3];

		if (Q >= 0)
		{
			D.x = uniform[off + 4];
			D.y = uniform[off + 5];
			Q = Q / (w * A);

			v = dot(w*D, pos) + phi * t;
		}
		else
		{
			float t_of_last_boop = (-Q);
			float t_ = t - t_of_last_boop;

			Q = 0.0f;

			math::float2 start(uniform[off + 4], uniform[off + 5]);

			math::float2 rel_pos = (pos - start);

			D = -normalize(rel_pos);

			Q = Q / (w * A);

			float wave_offset = 0.0f;
			v = dot(w*D, rel_pos) + phi * t_ + math::constants<float>::pi() + wave_offset;
			bool time_to_boop = v > (math::constants<float>::pi() + wave_offset) && v < (14 * math::constants<float>::pi());

			if (time_to_boop)
			{
				A = A / v;
				boop = 1.f;
			}
			else
			{
				A = 0.f;
			}
		}
		return boop;
	}

	__device__
	float addWavesPosition(int num_waves, float t, float mult_A, const math::float2& pos, math::float3& P) const
	{
		float boop = 0.f;
		for (int i = 0; i < num_waves; i++)
		{
			float Q, A, w, v;
			math::float2 D;

			boop = readWave(i, Q, A, D, w, v, mult_A, pos, t);

			P.x += Q * A*D.x*cos(v);
			P.y += Q * A*D.y*cos(v);
			P.z += A * sin(v);

		}
		return boop;
	}

	__device__
	float addWavesNormal(int num_waves, float t, float mult_A, const math::float2& pos, math::float3& B, math::float3& T) const
	{
		float boop = 0;
		for (int i = 0; i < num_waves; i++)
		{
			float Q, A, w, v;
			math::float2 D;

			boop = readWave(i, Q, A, D, w, v, mult_A, pos, t);

			float WA = w * A;
			float S = sin(v);
			float C = cos(v);

			B.x -= Q * D.x*D.x*WA*S;
			B.y -= Q * D.x*D.y*WA*S;
			B.z += D.x*WA*C;

			T.x -= Q * D.x*D.y*WA*S;
			T.y -= Q * D.y*D.y*WA*S;
			T.z += D.y*WA*C;
		}
		return boop;
	}

	__device__
	WaveVertex operator ()(const math::float4& v, math::float3& B, math::float3& T, math::float3& p, math::float2& tex_coord) const
	{
		float t = uniform[0];

		int num_waves = uniform[1];

		math::float2 pos(v.x + uniform[2], v.z + uniform[3]);

#ifdef CHECKER

		tex_coord = { pos.x, pos.y };

#else
		tex_coord = { 0.6f * pos.x, 0.6f * pos.y };
		tex_coord += t * math::float2{ 1, 0 } *0.0003f;
#endif

		float max_dist = 400.f;
		float mult_A = min(1.f, max_dist / (v.x*v.x + v.y*v.y + v.z*v.z));

		math::float3 P(pos.x, pos.y, 0);

		float posBoop = addWavesPosition(num_waves, t, mult_A, pos, P);

		B = { 1.f, 0, 0 };
		T = { 0, 1.f, 0 };

		pos.x = P.x;
		pos.y = P.y;

		float normalBoop = addWavesNormal(num_waves, t, mult_A, pos, B, T);

		B = B.xzy();
		T = T.xzy();
		p = P.xzy();

		return { camera.PVM * math::float4(p, 1.0f), p, normalize(cross(T, B)) };
	}
};

struct WaveQuadTriangulationShader
{
	__device__
	float operator ()(const WaveVertex& v1, const WaveVertex& v2, const WaveVertex& v3, const WaveVertex& v4)
	{
		//return abs(v1.p.y - v3.p.y);
		auto n1 = normalize(cross(v3.p - v2.p, v1.p - v2.p));
		auto n2 = normalize(cross(v1.p - v4.p, v3.p - v4.p));

		//return -min(min(min(min(min(dot(v1.n, n1), dot(v1.n, n2)), dot(v3.n, n1)), dot(v3.n, n2)), dot(v2.n, n1)), dot(v4.n, n2));
		//return (-dot(v1.n, n1) - dot(v1.n, n2) - dot(v3.n, n1) - dot(v3.n, n2) - dot(v2.n, n1) - dot(v4.n, n2)) * (1.0f / 6.0f);// (1.0f / (3.0f * (length(n1) + length(n2))));
		return -dot(v1.n, n1) - dot(v1.n, n2) - dot(v3.n, n1) - dot(v3.n, n2) - dot(v2.n, n1) - dot(v4.n, n2);
		//return fmaxf(v2.p.y, v4.p.y);
		//return dot(v1.n, n1) + dot(v1.n, n2) + dot(v3.n, n1) + dot(v3.n, n2) + dot(v2.n, n1) + dot(v4.n, n2);
	}
};

struct WaterFragmentShader : FragmentShader
{
	using FragmentShader::FragmentShader;

	__device__
	math::float4 operator ()(math::float3 B, math::float3 T, const math::float3& p, const math::float2& tex_coord) const
	{
#ifdef CHECKER
		float2 du = { 0.8f*dFdx(tex_coord.x), 0.8f*dFdy(tex_coord.x) };
		float2 dv = { 0.8f*dFdx(tex_coord.y), 0.8f*dFdy(tex_coord.y) };
		float4 tex_color_ = tex2DGrad(tex, tex_coord.x, tex_coord.y, du, dv);
		math::float3 tex_color = { tex_color_.x, tex_color_.y, tex_color_.z };

		math::float3 n = normalize(cross(T, B));

		math::float3 v = camera.position - p;

		float dist = sqrt(dot(v, v));

		v = normalize(v);

		float lambert = fmaxf(0, sqrt(dot(n, v)));

		math::float3 bg_vec(-v.x, -v.y, -v.z);

		bg_vec = normalize(bg_vec);

		math::float2 uv;
		uv.x = (1.f / (2.f * math::constants<float>::pi())) * (atan2(bg_vec.z, bg_vec.x) + math::constants<float>::pi());
		uv.y = 1.f - (2.f / (math::constants<float>::pi()) * asin(bg_vec.y));

		float4 bg_color = tex2D(texf, uv.x, uv.y);
		math::float3 background = { bg_color.x, bg_color.y, bg_color.z };

		float fade_factor = fmaxf(0.f, fminf(1.f, (dist - 50) / 50.f));
		fade_factor = 0;

		math::float3 value = fade_factor * tonemap(background) + (1.f - fade_factor) * tex_color;

		//math::float3 value = tex_color;

		return math::float4(value.x, value.y, value.z, 1.f);

#else
		math::float3 v = camera.position - p;

		float dist = sqrt(dot(v, v));

		v = normalize(v);

		math::float3 bg_vec(-v.x, -v.y, -v.z);

		bg_vec = normalize(bg_vec);

		math::float2 uv;
		uv.x = (1.f / (2.f * math::constants<float>::pi())) * (atan2(bg_vec.z, bg_vec.x) + math::constants<float>::pi());
		uv.y = 1.f - (2.f / (math::constants<float>::pi()) * asin(bg_vec.y));

		float4 bg_color = tex2D(texf, uv.x, uv.y);
		math::float3 background = { bg_color.x, bg_color.y, bg_color.z };

		//float fog_density = 0.1f;
		//float fog_factor = 1.f - exp(-fog_density * dist); 
		//fog_factor = fog_factor*fog_factor;
		float fade_factor = fmaxf(0.f, fminf(1.f, (dist - 50) / 50.f));
		//float fade_factor = 1.f;

		B = normalize(B);
		T = normalize(T);
		math::float3 n = normalize(cross(T, B));

		math::float3 orig_n = n;

		float2 du = { 0.5f * dFdx(tex_coord.x), 0.5f * dFdy(tex_coord.x) };
		float2 dv = { 0.5f * dFdx(tex_coord.y), 0.5f * dFdy(tex_coord.y) };
		float4 normal_lookup = tex2DGrad(tex, tex_coord.x, tex_coord.y, du, dv);

		math::float3 N = math::float3(normal_lookup.x, normal_lookup.y, normal_lookup.z) * 2 - 1.f;

		math::float3x3 mat(T.x, B.x, n.x,
			T.y, B.y, n.y,
			T.z, B.z, n.z);

		n = normalize(mat * N);

		math::float3 r0 = { 0.02f, 0.02f, 0.02f };
		float c = 1.f - dot(v, n);
		float c2 = c*c;
		float c5 = c2*c2*c;

		math::float3 fresnel = r0 + (1.f - r0) * c5;

		math::float3 r = 2 * dot(v, n) * n - v;

		uv.x = (1.f / (2.f * math::constants<float>::pi())) * (atan2(r.z, r.x) + math::constants<float>::pi());
		uv.y = 1.f - (2.f / (math::constants<float>::pi()) * asin(r.y));
		float4 tex_color = tex2D(texf, uv.x, uv.y);

		math::float3 reflection = { tex_color.x, tex_color.y, tex_color.z };

		math::float3 ocean_color = { 0.0f, 25 / 255.f, 34 / 255.f };

		float thinness = fmaxf(0, fminf(1, p.y*3.5f));
		float fac = thinness*thinness * fmaxf(0, dot(v, math::float3(orig_n.x, 0, orig_n.z)));

		//math::float3 color = 0.7f* fresnel * reflection + ((1 - fac) * ocean_color + fac * 2.f * ocean_color);
		math::float3 color = fresnel * reflection + ((1 - fac) * ocean_color + fac * 2.f * ocean_color);

		if (WIREFRAME)
		{
			float v = min(min(__f[0], __f[1]), __f[2]);

			if (v < 0.001f)
				//color = math::float3(0.f, 1.f, 1.f);
				color = math::float3(0.3f, 0.6f, 1.0f);
		}

		//math::float3 value = fade_factor * tonemap(background) + (1.f - fade_factor) * tonemap(color);
		math::float3 value = tonemap(fade_factor * background + (1.f - fade_factor) * color);

		return math::float4(value, 1.f);
#endif
	}
};

#endif  // INCLUDED_CURE_SHADERS_OCEAN
