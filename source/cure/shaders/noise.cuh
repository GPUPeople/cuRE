


#ifndef INCLUDED_CURE_SHADERS_NOISE
#define INCLUDED_CURE_SHADERS_NOISE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>
#include "uniforms.cuh"


namespace
{
	// based on Ken Perlin, Noise hardware. In Real-Time Shading SIGGRAPH Course Notes (2001), Olano M., (Ed.).
	// https://www.csee.umbc.edu/~olano/s2002c36/ch02.pdf
	//
	//template <int B>
	//__device__
	//constexpr int bit(int x)
	//{
	//	return (x >> B) & 1;
	//}
	//
	//template <int B>
	//__device__
	//constexpr unsigned int b(int i, int j, int k)
	//{
	//	constexpr unsigned int bit_patterns[] = { 0x15U, 0x38U, 0x32U, 0x2CU, 0x0DU, 0x13U, 0x07U, 0x2AU };
	//	unsigned int pattern_index = 4 * bit<B>(i) + 2 * bit<B>(j) + bit<B>(k);
	//	return bit_patterns[pattern_index & 0x3FU];
	//}
	//
	//__device__
	//constexpr math::float3 noiseGradient(int i, int j, int k)
	//{
	//	auto bits = b<0>(i, j, k) + b<1>(j, k, i) + b<2>(k, i, j) + b<3>(i, j, k) + b<4>(j, k, i) + b<5>(k, i, j) + b<6>(i, j, k) + b<7>(j, k, i);
	//	math::float3 p = (bits & 1U) ^ (bits & 2U) ? {1.0f, 0.0f, 0.0f} : {0.0f,1.0f,0.0f};
	//
	//	return g;
	//}


	// taken from https://en.wikipedia.org/wiki/Xorshift
	//
	__device__
	inline unsigned int rand(unsigned int x)
	{
		/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		return x;
	}

	__device__
	inline float randf(unsigned int x)
	{
		return __uint_as_float((127U << 23) | (rand(x) >> 9)) * 2.0f - 3.0f;
	}

	__device__
	inline math::float3 noiseGradient(const math::float3& p)
	{
		unsigned int seed1 = rand(__float_as_uint(p.x) ^ (__float_as_uint(p.z) << 16));
		unsigned int seed2 = rand(__float_as_uint(p.y) ^ (__float_as_uint(p.z) << 16));
		float u = randf(seed1);
		float theta = randf(seed2) * math::constants<float>::pi();

		float s = sqrt(1.0f - u * u);

		return { s * cos(theta), s * sin(theta), u };
	}

	__device__
	inline math::float3 skew(const math::float3& p)
	{
		float s = (p.x + p.y + p.z) / 3.0f;
		return { p.x + s, p.y + s, p.z + s };
	}

	__device__
	inline math::float3 unskew(const math::float3& p)
	{
		float s = (p.x + p.y + p.z) / 6.0f;
		return { p.x - s, p.y - s, p.z - s };
	}

	struct simplex_t
	{
		math::float3 v[4];
	};

	__device__
	inline simplex_t simplex(const math::float3& p)
	{
		auto skewed_base = floor(skew(p));
		auto v_0 = unskew(skewed_base);

		auto v = p - v_0;

		bool b1 = v.x < v.y;
		bool b2 = v.y < v.z;
		bool b3 = v.z < v.x;

		math::float3 e_1 = { ~b1 & b3 ? 1.0f : 0.0f, ~b2 & b1 ? 1.0f : 0.0f, ~b3 & b2 ? 1.0f : 0.0f };
		math::float3 e_2 = { b1 ^ b3 ? 0.0f : 1.0f, b2 ^ b1 ? 0.0f : 1.0f, b3 ^ b2 ? 0.0f : 1.0f };
		math::float3 e_3 = { b1 & ~b3 ? 1.0f : 0.0f, b2 & ~b1 ? 1.0f : 0.0f, b3 & ~b2 ? 1.0f : 0.0f };

		return { v_0, v_0 + unskew(e_1), v_0 + unskew(e_1 + e_2), v_0 + unskew(e_1 + e_2 + e_3) };
	}

	__device__
	inline float simplexContribution(const math::float3& d, const math::float3& gradient)
	{
		float t = 0.5f - dot(d, d);
		float t2 = t * t;
		float t4 = t2 * t2;
		return t > 0.0f ? 8.0f * t4 * dot(gradient, d) : 0.0f;
	}
}

__device__
inline float simplexNoise(const math::float3& p)
{
	auto s = simplex(p);

	return
		simplexContribution(p - s.v[0], noiseGradient(s.v[0])) +
		simplexContribution(p - s.v[1], noiseGradient(s.v[1])) +
		simplexContribution(p - s.v[2], noiseGradient(s.v[2])) +
		simplexContribution(p - s.v[3], noiseGradient(s.v[3]));
}


__device__
inline float simplexNoiseFractal(const math::float3& p)
{
	const int N = __float_as_int(uniform[5]);

	float r = 0.0f;

	for (int i = 0; i < N; ++i)
		r += 1.0f / (1 << i) * simplexNoise(p * (1 << i));

	return 1.0f / (2.0f - 1.0f / (1 << (N - 1))) * r;
}

#endif  // INCLUDED_CURE_SHADERS_NOISE
