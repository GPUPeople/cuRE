


#ifndef INCLUDED_CURE_FRAGMENT_SHADER_INPUT
#define INCLUDED_CURE_FRAGMENT_SHADER_INPUT

#pragma once

#include <math/vector.h>

#include "shader.cuh"


template <typename T>
struct FragmentShaderInput;

template <>
struct FragmentShaderInput<float>
{
	template <int CURRENT_INTERPOLATOR, int CURRENT_LOCATION>
	struct Pack
	{
		static constexpr int size = 1;
		static constexpr int interpolator = CURRENT_LOCATION > 3 ? CURRENT_INTERPOLATOR + 1 : CURRENT_INTERPOLATOR;
		static constexpr int location = interpolator != CURRENT_INTERPOLATOR ? 0 : CURRENT_LOCATION;
	};

	template <unsigned int location>
	__device__
	static float load(const math::float4& interpolator);

	template <>
	__device__
	static float load<0>(const math::float4& interpolator)
	{
		return interpolator.x;
	}

	template <>
	__device__
	static float load<1>(const math::float4& interpolator)
	{
		return interpolator.y;
	}

	template <>
	__device__
	static float load<2>(const math::float4& interpolator)
	{
		return interpolator.z;
	}

	template <>
	__device__
	static float load<3>(const math::float4& interpolator)
	{
		return interpolator.w;
	}

	template <unsigned int location>
	__device__
	static void store(math::float4& interpolator, float value);

	template <>
	__device__
	static void store<0>(math::float4& interpolator, float value)
	{
		interpolator.x = value;
	}

	template <>
	__device__
	static void store<1>(math::float4& interpolator, float value)
	{
		interpolator.y = value;
	}

	template <>
	__device__
	static void store<2>(math::float4& interpolator, float value)
	{
		interpolator.z = value;
	}

	template <>
	__device__
	static void store<3>(math::float4& interpolator, float value)
	{
		interpolator.w = value;
	}
};

template <>
struct FragmentShaderInput<math::float2>
{
	template <int CURRENT_INTERPOLATOR, int CURRENT_LOCATION>
	struct Pack
	{
		static constexpr int size = 2;
		static constexpr int interpolator = CURRENT_LOCATION > 2 ? CURRENT_INTERPOLATOR + 1 : CURRENT_INTERPOLATOR;
		static constexpr int location = interpolator != CURRENT_INTERPOLATOR ? 0 : CURRENT_LOCATION;
	};

	template <unsigned int location>
	__device__
	static math::float2 load(const math::float4& interpolator);

	template <>
	__device__
	static math::float2 load<0>(const math::float4& interpolator)
	{
		return math::float2(interpolator.x, interpolator.y);
	}

	template <>
	__device__
	static math::float2 load<1>(const math::float4& interpolator)
	{
		return math::float2(interpolator.y, interpolator.z);
	}

	template <>
	__device__
	static math::float2 load<2>(const math::float4& interpolator)
	{
		return math::float2(interpolator.z, interpolator.w);
	}

	template <unsigned int location>
	__device__
	static void store(math::float4& interpolator, const math::float2& value);

	template <>
	__device__
	static void store<0>(math::float4& interpolator, const math::float2& value)
	{
		interpolator.x = value.x; interpolator.y = value.y;
	}

	template <>
	__device__
	static void store<1>(math::float4& interpolator, const math::float2& value)
	{
		interpolator.y = value.x; interpolator.z = value.y;
	}

	template <>
	__device__
	static void store<2>(math::float4& interpolator, const math::float2& value)
	{
		interpolator.z = value.x; interpolator.w = value.y;
	}
};

template <>
struct FragmentShaderInput<math::float3>
{
	template <int CURRENT_INTERPOLATOR, int CURRENT_LOCATION>
	struct Pack
	{
		static constexpr int size = 3;
		static constexpr int interpolator = CURRENT_LOCATION > 1 ? CURRENT_INTERPOLATOR + 1 : CURRENT_INTERPOLATOR;
		static constexpr int location = interpolator != CURRENT_INTERPOLATOR ? 0 : CURRENT_LOCATION;
	};

	template <unsigned int location>
	__device__
	static math::float3 load(const math::float4& interpolator);

	template <>
	__device__
	static math::float3 load<0>(const math::float4& interpolator)
	{
		return math::float3(interpolator.x, interpolator.y, interpolator.z);
	}

	template <>
	__device__
	static math::float3 load<1>(const math::float4& interpolator)
	{
		return math::float3(interpolator.y, interpolator.z, interpolator.w);
	}

	template <unsigned int location>
	__device__
	static void store(math::float4& interpolator, const math::float3& value);

	template <>
	__device__
	static void store<0>(math::float4& interpolator, const math::float3& value)
	{
		interpolator.x = value.x; interpolator.y = value.y; interpolator.z = value.z;
	}

	template <>
	__device__
	static void store<1>(math::float4& interpolator, const math::float3& value)
	{
		interpolator.y = value.x; interpolator.z = value.y; interpolator.w = value.z;
	}
};

template <>
struct FragmentShaderInput<math::float4>
{
	template <int CURRENT_INTERPOLATOR, int CURRENT_LOCATION>
	struct Pack
	{
		static constexpr int size = 4;
		static constexpr int interpolator = CURRENT_LOCATION != 0 ? CURRENT_INTERPOLATOR + 1 : CURRENT_INTERPOLATOR;
		static constexpr int location = 0;
	};

	template <unsigned int location>
	__device__
	static math::float4 load(const math::float4& interpolator);

	template <>
	__device__
	static math::float4 load<0>(const math::float4& interpolator)
	{
		return interpolator;
	}

	template <unsigned int location>
	__device__
	static void store(math::float4& interpolator, const math::float4& value);

	template <>
	__device__
	static void store<0>(math::float4& interpolator, const math::float4& value)
	{
		interpolator = value;
	}
};



template <int START_INTERPOLATOR, int START_LOCATION, typename... S>
struct Interpolators;

template <>
struct Interpolators<0, 0>
{
	static constexpr int count = 0;

	template <typename F, typename I>
	__device__
	static auto read(F& reader, const I& interpolators)
	{
		return reader();
	}

	template <typename F, typename I>
	__device__
	static auto write(F& writer, I& interpolators)
	{
		return writer();
	}
};

template <int START_INTERPOLATOR, int START_LOCATION>
struct Interpolators<START_INTERPOLATOR, START_LOCATION>
{
	static constexpr int count = START_INTERPOLATOR + 1;

	template <typename F, typename I, typename... Args>
	__device__
	static auto read(F& reader, const I& interpolators, Args&&... args)
	{
		return reader(args...);
	}

	template <typename F, typename I, typename... Args>
	__device__
	static auto write(F& writer, I& interpolators, Args&... args)
	{
		return writer(args...);
	}
};

template <int START_INTERPOLATOR, int START_LOCATION, typename T, typename... Tail>
struct Interpolators<START_INTERPOLATOR, START_LOCATION, T, Tail...>
{
	typedef FragmentShaderInput<T>::template Pack<START_INTERPOLATOR, START_LOCATION> Packing;
	typedef Interpolators<Packing::interpolator, Packing::location + Packing::size, Tail...> Next;
	static constexpr int count = Next::count;


	template <typename F, typename I, typename... Args>
	__device__
	static auto read(F& reader, const I& interpolators, Args&&... args)
	{
		return Next::read(reader, interpolators, args..., FragmentShaderInput<T>::template load<Packing::location>(interpolators[Packing::interpolator]));
	}

	template <typename F, typename I, typename... Args>
	__device__
	static auto write(F& writer, I& interpolators, Args&... args)
	{
		T value;
		auto ret = Next::write(writer, interpolators, args..., value);
		FragmentShaderInput<T>::template store<Packing::location>(interpolators[Packing::interpolator], value);
		return ret;
	}
};

#endif  // INCLUDED_CURE_FRAGMENT_SHADER_INPUT
