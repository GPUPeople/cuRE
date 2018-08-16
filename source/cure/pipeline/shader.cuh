


#ifndef INCLUDED_CURE_SHADER
#define INCLUDED_CURE_SHADER

#pragma once

#include <math/vector.h>

#include <meta_utils.h>


template <typename T>
struct ShaderInputType
{
	typedef T type;
};

template <typename T>
struct ShaderInputType<const T&>
{
	typedef T type;
};

template <typename T>
struct ShaderInputType<T&> {};


template <typename T>
struct ShaderOutputType {};

template <typename T>
struct ShaderOutputType<const T&> {};

template <typename T>
struct ShaderOutputType<T&>
{
	typedef T type;
};



template <typename... S>
struct ShaderSignature
{
};


namespace internal
{
	template <typename I, typename O, typename = void, typename... Tail>
	struct SignatureBuilder;

	template <typename... I, typename... O>
	struct SignatureBuilder<ShaderSignature<I...>, ShaderSignature<O...>, void>
	{
		typedef ShaderSignature<I...> In;
		typedef ShaderSignature<O...> Out;
	};

	template <typename... I, typename... O, typename T, typename... Tail>
	struct SignatureBuilder<ShaderSignature<I...>, ShaderSignature<O...>, typename void_t<typename ShaderInputType<T>::type>::type, T, Tail...>
	  : SignatureBuilder<ShaderSignature<I..., typename ShaderInputType<T>::type>, ShaderSignature<O...>, void, Tail...>
	{
	};

	template <typename... I, typename... O, typename T, typename... Tail>
	struct SignatureBuilder<ShaderSignature<I...>, ShaderSignature<O...>, typename void_t<typename ShaderOutputType<T>::type>::type, T, Tail...>
	  : SignatureBuilder<ShaderSignature<I...>, ShaderSignature<O..., typename ShaderOutputType<T>::type>, void, Tail...>
	{
	};
}


template <typename VS>
struct VertexShaderInfo;

template <class Ret, typename... A>
struct VertexShaderInfo<Ret(A...)>
{
private:
	typedef internal::SignatureBuilder<ShaderSignature<>, ShaderSignature<>, void, A...> Signature;

public:
	using Inputs = typename Signature::In;
	using Outputs = typename Signature::Out;
	using Args = ShaderSignature<A...>;
};

template <class Ret, typename... Args>
struct VertexShaderInfo<Ret(&)(Args...)> : VertexShaderInfo<Ret(Args...)>
{
};

template <class Ret, typename... Args>
struct VertexShaderInfo<Ret(*)(Args...)> : VertexShaderInfo<Ret(Args...)>
{
};

template <class Ret, class C, typename... Args>
struct VertexShaderInfo<Ret(C::*)(Args...) const> : VertexShaderInfo<Ret(Args...)>
{
};

template <class VS>
struct VertexShaderInfo : VertexShaderInfo<decltype(&VS::operator ())>
{
};


template <typename FS>
struct FragmentShaderInfo;

template <class Ret, typename... Args>
struct FragmentShaderInfo<Ret(Args...)>
{
private:
	typedef internal::SignatureBuilder<ShaderSignature<>, ShaderSignature<>, void, Args...> Signature;

public:
	typedef Signature::In Inputs;
	typedef Signature::Out Outputs;
};

template <class Ret, typename... Args>
struct FragmentShaderInfo<Ret(&)(Args...)> : FragmentShaderInfo<Ret(Args...)>
{
};

template <class Ret, typename... Args>
struct FragmentShaderInfo<Ret(*)(Args...)> : FragmentShaderInfo<Ret(Args...)>
{
};

template <class Ret, class C, typename... Args>
struct FragmentShaderInfo<Ret(C::*)(Args...) const> : FragmentShaderInfo<Ret(Args...)>
{
};

template <class Ret, class C, typename... Args>
struct FragmentShaderInfo<Ret(C::*)(Args...)> : FragmentShaderInfo<Ret(Args...)>
{
};

template <class FS>
struct FragmentShaderInfo : FragmentShaderInfo<decltype(&FS::operator ())>
{
};

#endif  // INCLUDED_CURE_SHADER
