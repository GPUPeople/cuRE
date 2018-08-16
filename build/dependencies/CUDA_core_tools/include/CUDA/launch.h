


#ifndef INCLUDED_CUDA_LAUNCH
#define INCLUDED_CUDA_LAUNCH

#pragma once

#include <type_traits>
#include <utility>
#include <cstddef>
#include <cstring>
#include <cstdlib>

#include <cuda.h>

#include "type_traits.h"
#include "error.h"


namespace CU
{
	template <typename T>
	inline constexpr std::enable_if_t<std::is_unsigned<T>::value, T> divup(T a, T b)
	{
		return (a + b - 1) / b;
	}


	class dim
	{
		unsigned int d[3];

	public:
		dim(unsigned int d0, unsigned int d1 = 1U, unsigned int d2 = 1U)
			: d {d0, d1, d2}
		{
		}

		const unsigned int& operator [](int i) const { return d[i]; }
		unsigned int& operator [](int i) { return d[i]; }
	};


	template <std::size_t offset, std::size_t alignment, typename... Args>
	class ArgumentBuffer;

	template <std::size_t offset, std::size_t alignment>
	class ArgumentBuffer<offset, alignment>
	{
		alignas(alignment) char buffer[offset];
	public:
		char* data() { return buffer; }
	};

	template <std::size_t offset, std::size_t alignment, typename A, typename... Tail>
	class ArgumentBuffer<offset, alignment, A, Tail...>
	{
		static_assert(std::is_trivially_copyable<A>::value, "argument type must be trivially copyable");

		static constexpr std::size_t start = divup(offset, alignment_of<A>::value) * alignment_of<A>::value;

		ArgumentBuffer<start + sizeof(A), alignment < alignment_of<A>::value ? alignment_of<A>::value : alignment, Tail...> tail;

	public:
		ArgumentBuffer(const A& arg, const Tail&... tailargs)
			: tail(tailargs...)
		{
			std::memcpy(tail.data() + start, &arg, sizeof(A));
		}

		char* data() { return tail.data(); }
	};


	template <typename... Args>
	inline void call(CUfunction f, dim grid_dim, dim block_dim, unsigned int shared_memory, CUstream stream, Args&&... args)
	{
		ArgumentBuffer<0U, 0U, std::remove_reference_t<Args>...> arg_buffer { std::forward<Args>(args)... };
		std::size_t arg_buffer_size = sizeof(arg_buffer);
		
		void* options[] = {
			CU_LAUNCH_PARAM_BUFFER_POINTER, &arg_buffer,
			CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_buffer_size,
			CU_LAUNCH_PARAM_END
		};
		
		succeed(cuLaunchKernel(f, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0], block_dim[1], block_dim[2], shared_memory, stream, nullptr, options));
	}

	template <>
	inline void call<>(CUfunction f, dim grid_dim, dim block_dim, unsigned int shared_memory, CUstream stream)
	{
		succeed(cuLaunchKernel(f, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0], block_dim[1], block_dim[2], shared_memory, stream, nullptr, nullptr));
	}

	template <typename... Args>
	class Function
	{
		CUfunction f;

	public:
		explicit Function(CUfunction f)
			: f(f)
		{
		}

		operator CUfunction() const { return f; }

		void operator ()(dim grid_dim, dim block_dim, unsigned int shared_memory, CUstream stream, device_type_t<Args>... args) const
		{
			call(f, grid_dim, block_dim, shared_memory, stream, args...);
		}
	};


	template <typename... Args>
	inline void callCooperative(CUfunction f, dim grid_dim, dim block_dim, unsigned int shared_memory, CUstream stream, Args... args)
	{
		void* kernel_args[] = { &args... };
		succeed(cuLaunchCooperativeKernel(f, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0], block_dim[1], block_dim[2], shared_memory, stream, kernel_args));
	}

	template <>
	inline void callCooperative<>(CUfunction f, dim grid_dim, dim block_dim, unsigned int shared_memory, CUstream stream)
	{
		succeed(cuLaunchCooperativeKernel(f, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0], block_dim[1], block_dim[2], shared_memory, stream, nullptr));
	}

	template <typename... Args>
	class CooperativeFunction
	{
		CUfunction f;

	public:
		explicit CooperativeFunction(CUfunction f)
			: f(f)
		{
		}

		operator CUfunction() const { return f; }

		void operator ()(dim grid_dim, dim block_dim, unsigned int shared_memory, CUstream stream, device_type_t<Args>... args) const
		{
			callCooperative(f, grid_dim, block_dim, shared_memory, stream, args...);
		}
	};
}

#endif  // INCLUDED_CUDA_LAUNCH
