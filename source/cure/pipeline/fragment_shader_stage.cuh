


#ifndef INCLUDED_CURE_FRAGMENT_SHADER_STAGE
#define INCLUDED_CURE_FRAGMENT_SHADER_STAGE

#pragma once

#include <math/vector.h>

#include "shader.cuh"
#include "fragment_shader_input.cuh"



class CoverageShader
{
protected:
	const math::int4 __tile_bounds;
	const float __left;
	const float __top;
	const float __right;
	const float __bottom;

public:
	__device__
	CoverageShader(const math::int4& tile_bounds, float left, float top, float right, float bottom)
		: __tile_bounds(tile_bounds), __left(left), __top(top), __right(right), __bottom(bottom)
	{
	}

	template <typename Bitmask>
	__device__
	Bitmask operator()(Bitmask mask) const
	{
		return mask;
	}
};


struct FragmentShader
{
	bool discard_fragment = false;

protected:
	const int __triangleID;
	const math::int2 __pixel_loc;
	const math::float4 __frag_coord;
	const math::float3 __lambda;
	const math::float3 __f;

	__device__
	float dFdx(float v) const
	{
		if (!FORCE_QUAD_SHADING)
			return 0.0f;

		float xother = __shfl_sync(~0U, v, threadIdx.x + 1, 2);
		float dx = (xother - v);
		dx = __uint_as_float(__float_as_uint(dx) ^ ((threadIdx.x & 0x1U) << 31U));
		//dx = dx *((threadIdx.x & 0x1) == 0x1 ? -1.0f : 1.0f);
		return dx;// / (0.5f * pixel_scale.x);
	}

	__device__
	float dFdy(float v) const
	{
		if (!FORCE_QUAD_SHADING)
			return 0.0f;

		float yother = __shfl_sync(~0U, v, threadIdx.x + 2, 4);
		float dy = (yother - v);
		//dy = dy*((threadIdx.x & 0x2) == 0x2 ? -1.0f : 1.0f);
		dy = __uint_as_float(__float_as_uint(dy) ^ ((threadIdx.x & 0x2U) << 30U));
		return dy;// / (0.5f * pixel_scale.y);
	}

	__device__
	void discard()
	{
		discard_fragment = true;
	}

public:
	__device__
	FragmentShader(const math::int2& pixel_loc, const math::float4& frag_coord, const math::float3& lambda, const math::float3& f, int triangleID)
		: __pixel_loc(pixel_loc), __frag_coord(frag_coord), __lambda(lambda), __f(f), __triangleID(triangleID)
	{
	}

	__device__
	bool discarded() const
	{
		return discard_fragment;
	}
};


template <typename FS>
class FragmentShaderInputStorage;

template <>
class FragmentShaderInputStorage<ShaderSignature<>>
{
public:
	template <class TriangleBuffer>
	__device__
	void loadWarp(const TriangleBuffer& triangle_buffer, unsigned int triangle_id)
	{
	}

	template <class TriangleBuffer>
	__device__
	void load(const TriangleBuffer& triangle_buffer, unsigned int triangle_id)
	{
	}

	template <typename F>
	__device__
	auto read(F& reader, const math::float3& lambda) const
	{
		return reader();
	}
};

template <typename... S>
class FragmentShaderInputStorage<ShaderSignature<S...>>
{
private:
	typedef ::Interpolators<0, 0, S...> Interpolators;

	math::float4x3 interpolators[Interpolators::count];

public:

	template <class TriangleBuffer>
	__device__
	void load(const TriangleBuffer& triangle_buffer, unsigned int triangle_id)
	{
		triangle_buffer.loadInterpolator(interpolators, Interpolators::count, triangle_id);
	}

	template <class TriangleBuffer>
	__device__
	void loadWarp(const TriangleBuffer& triangle_buffer, unsigned int triangle_id)
	{
		triangle_buffer.loadInterpolatorsWarp(interpolators, Interpolators::count, triangle_id);
	}

	template <typename F>
	__device__
	auto read(F& reader, const math::float3& lambda) const
	{
		math::float4 interpolated[Interpolators::count];

		for (int i = 0; i < Interpolators::count; ++i)
			interpolated[i] = interpolators[i] * lambda;

		return Interpolators::read([&](const S&... args)
		{
			return reader(args...);
		}, interpolated);
	}
};


template <typename FS, class FragmentShaderInputStorage>
__device__
inline math::float4 callFragmentShader(FS& fs, const FragmentShaderInputStorage& fs_input, const math::float3& lambda)
{
	return fs_input.read(fs, lambda);
}

#endif  // INCLUDED_CURE_FRAGMENT_SHADER_STAGE
