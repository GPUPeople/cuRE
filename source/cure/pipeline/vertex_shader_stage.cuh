


#ifndef INCLUDED_CURE_VERTEX_SHADER_STAGE
#define INCLUDED_CURE_VERTEX_SHADER_STAGE

#pragma once

#include <math/vector.h>

#include "shader.cuh"
#include "vertex_shader_input.cuh"
#include "fragment_shader_input.cuh"


template <typename S>
class VertexShaderOutputStorage;


template <>
class VertexShaderOutputStorage<ShaderSignature<>>
{
public:
	using Signature = ShaderSignature<>;

	static constexpr int NUM_INTERPOLATORS = 0;

	template <typename F>
	__device__
	auto write(F& writer)
	{
		return writer();
	}

	template <class TriangleBuffer>
	__device__
	friend void store(TriangleBuffer& triangle_buffer, unsigned int triangle_id, const VertexShaderOutputStorage& o1, const VertexShaderOutputStorage& o2, const VertexShaderOutputStorage& o3)
	{
	}
};

template <typename... S>
class VertexShaderOutputStorage<ShaderSignature<S...>>
{
	typedef ::Interpolators<0, 0, S...> Interpolators;

	math::float4 interpolators[Interpolators::count];

public:
	using Signature = ShaderSignature<S...>;

	static constexpr int NUM_INTERPOLATORS = Interpolators::count;

	template <typename F>
	__device__
	auto write(F& writer)
	{
		return Interpolators::write([&](S&... args)
		{
			return writer(args...);
		}, interpolators);
	}

	template <class TriangleBuffer>
	__device__
	friend void store(TriangleBuffer& triangle_buffer, unsigned int triangle_id, const VertexShaderOutputStorage& o1, const VertexShaderOutputStorage& o2, const VertexShaderOutputStorage& o3)
	{
		#pragma unroll
		for (int j = 0; j < Interpolators::count; ++j)
			triangle_buffer.storeInterpolator(j, triangle_id, math::float4x3::from_cols(o1.interpolators[j], o2.interpolators[j], o3.interpolators[j]));
	}
};


namespace internal
{
	template <typename In, typename Out, typename VSArgs, typename = void>
	struct VertexShaderCall;


	template <>
	struct VertexShaderCall<ShaderSignature<>, ShaderSignature<>, ShaderSignature<>, void>
	{
		template <typename VS, typename... Args>
		__device__
		static auto call(VS& shader, Args&&... args)
		{
			return shader(args...);
		}
	};

	template <typename... In, typename T, typename... VSArgs>
	struct VertexShaderCall<ShaderSignature<In...>, ShaderSignature<>, ShaderSignature<T, VSArgs...>, typename void_t<typename ShaderOutputType<T>::type>::type>
	{
		template <typename VS, typename... Args>
		__device__
		static auto call(VS& shader, const In&... in, Args&&... args)
		{
			typename ShaderOutputType<T>::type o;
			return VertexShaderCall<ShaderSignature<In...>, ShaderSignature<>, ShaderSignature<VSArgs...>>::call(shader, in..., args..., o);
		}
	};

	template <typename... Out, typename T, typename... VSArgs>
	struct VertexShaderCall<ShaderSignature<>, ShaderSignature<Out...>, ShaderSignature<T, VSArgs...>, typename void_t<typename ShaderInputType<T>::type>::type>
	{
		template <typename VS, typename... Args>
		__device__
		static auto call(VS& shader, Out&... out, Args&&... args)
		{
			return VertexShaderCall<ShaderSignature<>, ShaderSignature<Out...>, ShaderSignature<VSArgs...>>::call(shader, out..., args..., typename ShaderInputType<T>::type());
		}
	};

	template <typename... In, typename... Out, typename T, typename... VSArgs>
	struct VertexShaderCall<ShaderSignature<typename ShaderInputType<T>::type, In...>, ShaderSignature<Out...>, ShaderSignature<T, VSArgs...>, void>
	{
		typedef typename ShaderInputType<T>::type I;

		template <typename VS, typename... Args>
		__device__
		static auto call(VS& shader, Out&... out, const I& i, const In&... in, Args&&... args)
		{
			return VertexShaderCall<ShaderSignature<In...>, ShaderSignature<Out...>, ShaderSignature<VSArgs...>>::call(shader, out..., in..., args..., i);
		}
	};

	template <typename... In, typename... Out, typename T, typename... VSArgs>
	struct VertexShaderCall<ShaderSignature<In...>, ShaderSignature<typename ShaderOutputType<T>::type, Out...>, ShaderSignature<T, VSArgs...>, void>
	{
		typedef typename ShaderOutputType<T>::type O;

		template <typename VS, typename... Args>
		__device__
		static auto call(VS& shader, O& o, Out&... out, const In&... in, Args&&... args)
		{
			return VertexShaderCall<ShaderSignature<In...>, ShaderSignature<Out...>, ShaderSignature<VSArgs...>>::call(shader, out..., in..., args..., o);
		}
	};
}


//template <typename VS, typename O, typename I>
//__device__
//inline math::float4 callVertexShader(VS shader, O& vs_output, const I& vs_input)
//{
//	using Args = typename VertexShaderInfo<VS>::Args;
//
//	return vs_output.write([&](auto&... out)
//	{
//		return vs_input.read([&](const auto&... in)
//		{
//			return internal::VertexShaderCall<typename I::Signature, typename O::Signature, Args>::call(shader, out..., in...);
//		});
//	});
//}

template <typename VS, typename... Out, typename VB, typename... In>
__device__
inline auto callVertexShader(VS& shader, VertexShaderOutputStorage<ShaderSignature<Out...>>& vs_output, const InputVertexAttributes<VertexBufferAttributes<VB, In...>>& vs_input)
{
	typedef typename VertexShaderInfo<VS>::Args Args;

	return vs_output.write([&](Out&... out)
	{
		return vs_input.read([&](const In&... in)
		{
			return internal::VertexShaderCall<ShaderSignature<In...>, ShaderSignature<Out...>, Args>::call(shader, out..., in...);
		});
	});
}

#endif  // INCLUDED_CURE_VERTEX_SHADER_STAGE
