


#ifndef INCLUDED_CUDA_ERROR
#define INCLUDED_CUDA_ERROR

#pragma once

#include <type_traits>
#include <exception>

#include <cuda.h>


namespace CU
{
	template <CUresult error_code, typename = void>
	struct error_traits;

	template <CUresult error_code>
	class basic_error : public error_traits<error_code>::category
	{
	public:
		virtual CUresult code() const noexcept override;
		virtual const char* name() const noexcept override;
		const char* what() const noexcept override;
	};
	

	class error : public std::exception
	{
	public:
		virtual CUresult code() const noexcept = 0;
		virtual const char* name() const noexcept = 0;

		using invalid_value = basic_error<CUDA_ERROR_INVALID_VALUE>;
		using out_of_memory = basic_error<CUDA_ERROR_OUT_OF_MEMORY>;
		using not_initialized = basic_error<CUDA_ERROR_NOT_INITIALIZED>;
		using deinitialized = basic_error<CUDA_ERROR_DEINITIALIZED>;
		using profiler_disabled = basic_error<CUDA_ERROR_PROFILER_DISABLED>;
		//using profiler_not_initialized = basic_error<CUDA_ERROR_PROFILER_NOT_INITIALIZED>;
		//using profiler_already_started = basic_error<CUDA_ERROR_PROFILER_ALREADY_STARTED>;
		//using profiler_already_stopped = basic_error<CUDA_ERROR_PROFILER_ALREADY_STOPPED>;
		using no_device = basic_error<CUDA_ERROR_NO_DEVICE>;
		using invalid_device = basic_error<CUDA_ERROR_INVALID_DEVICE>;
		using invalid_image = basic_error<CUDA_ERROR_INVALID_IMAGE>;
		using invalid_context = basic_error<CUDA_ERROR_INVALID_CONTEXT>;
		//using context_already_current = basic_error<CUDA_ERROR_CONTEXT_ALREADY_CURRENT>;
		using map_failed = basic_error<CUDA_ERROR_MAP_FAILED>;
		using unmap_failed = basic_error<CUDA_ERROR_UNMAP_FAILED>;
		using array_is_mapped = basic_error<CUDA_ERROR_ARRAY_IS_MAPPED>;
		using already_mapped = basic_error<CUDA_ERROR_ALREADY_MAPPED>;
		using no_binary_for_gpu = basic_error<CUDA_ERROR_NO_BINARY_FOR_GPU>;
		using already_acquired = basic_error<CUDA_ERROR_ALREADY_ACQUIRED>;
		using not_mapped = basic_error<CUDA_ERROR_NOT_MAPPED>;
		using not_mapped_as_array = basic_error<CUDA_ERROR_NOT_MAPPED_AS_ARRAY>;
		using not_mapped_as_pointer = basic_error<CUDA_ERROR_NOT_MAPPED_AS_POINTER>;
		using ecc_uncorrectable = basic_error<CUDA_ERROR_ECC_UNCORRECTABLE>;
		using unsupported_limit = basic_error<CUDA_ERROR_UNSUPPORTED_LIMIT>;
		using context_already_in_use = basic_error<CUDA_ERROR_CONTEXT_ALREADY_IN_USE>;
		using peer_access_unsupported = basic_error<CUDA_ERROR_PEER_ACCESS_UNSUPPORTED>;
		using invalid_ptx = basic_error<CUDA_ERROR_INVALID_PTX>;
		using invalid_graphics_context = basic_error<CUDA_ERROR_INVALID_GRAPHICS_CONTEXT>;
		using nvlink_uncorrectable = basic_error<CUDA_ERROR_NVLINK_UNCORRECTABLE>;
		using jit_not_found = basic_error<CUDA_ERROR_JIT_COMPILER_NOT_FOUND>;
		using invalid_source = basic_error<CUDA_ERROR_INVALID_SOURCE>;
		using file_not_found = basic_error<CUDA_ERROR_FILE_NOT_FOUND>;
		using shared_object_symbol_not_found = basic_error<CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND>;
		using shared_object_init_failed = basic_error<CUDA_ERROR_SHARED_OBJECT_INIT_FAILED>;
		using operating_system = basic_error<CUDA_ERROR_OPERATING_SYSTEM>;
		using invalid_handle = basic_error<CUDA_ERROR_INVALID_HANDLE>;
		using not_found = basic_error<CUDA_ERROR_NOT_FOUND>;
		using not_ready = basic_error<CUDA_ERROR_NOT_READY>;
		using illegal_address = basic_error<CUDA_ERROR_ILLEGAL_ADDRESS>;
		using launch_out_of_resources = basic_error<CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES>;
		using launch_timeout = basic_error<CUDA_ERROR_LAUNCH_TIMEOUT>;
		using launch_incompatible_texturing = basic_error<CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING>;
		using peer_access_already_enabled = basic_error<CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED>;
		using peer_access_not_enabled = basic_error<CUDA_ERROR_PEER_ACCESS_NOT_ENABLED>;
		using primary_context_active = basic_error<CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE>;
		using context_is_destroyed = basic_error<CUDA_ERROR_CONTEXT_IS_DESTROYED>;
		using assertion_failed = basic_error<CUDA_ERROR_ASSERT>;
		using too_many_peers = basic_error<CUDA_ERROR_TOO_MANY_PEERS>;
		using host_memory_already_registered = basic_error<CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED>;
		using host_memory_not_registered = basic_error<CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED>;
		using hardware_stack_error = basic_error<CUDA_ERROR_HARDWARE_STACK_ERROR>;
		using illegal_instruction = basic_error<CUDA_ERROR_ILLEGAL_INSTRUCTION>;
		using misaligned_address = basic_error<CUDA_ERROR_MISALIGNED_ADDRESS>;
		using invalid_address_space = basic_error<CUDA_ERROR_INVALID_ADDRESS_SPACE>;
		using invalid_pc = basic_error<CUDA_ERROR_INVALID_PC>;
		using launch_failed = basic_error<CUDA_ERROR_LAUNCH_FAILED>;
		using cooperative_launch_too_large = basic_error<CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE>;
		using not_permitted = basic_error<CUDA_ERROR_NOT_PERMITTED>;
		using not_supported = basic_error<CUDA_ERROR_NOT_SUPPORTED>;
		using unknown = basic_error<CUDA_ERROR_UNKNOWN>;
	};

	class logic_error : public error {};
	class runtime_error : public error {};
	class fatal_error : public runtime_error {};
	class bad_alloc : public error {};


	template <CUresult error_code, typename>
	struct error_traits
	{
		using category = logic_error;
	};

	template <CUresult error_code>
	struct error_traits<error_code, std::enable_if_t<
		error_code == CUDA_ERROR_PROFILER_DISABLED ||
		error_code == CUDA_ERROR_MAP_FAILED ||
		error_code == CUDA_ERROR_UNMAP_FAILED ||
		error_code == CUDA_ERROR_ECC_UNCORRECTABLE ||
		error_code == CUDA_ERROR_NVLINK_UNCORRECTABLE ||
		error_code == CUDA_ERROR_JIT_COMPILER_NOT_FOUND ||
		error_code == CUDA_ERROR_OPERATING_SYSTEM
		>>
	{
		using category = runtime_error;
	};

	template <CUresult error_code>
	struct error_traits<error_code, std::enable_if_t<
		error_code == CUDA_ERROR_ILLEGAL_ADDRESS ||
		error_code == CUDA_ERROR_LAUNCH_TIMEOUT ||
		error_code == CUDA_ERROR_ASSERT ||
		error_code == CUDA_ERROR_HARDWARE_STACK_ERROR ||
		error_code == CUDA_ERROR_ILLEGAL_INSTRUCTION ||
		error_code == CUDA_ERROR_MISALIGNED_ADDRESS ||
		error_code == CUDA_ERROR_INVALID_ADDRESS_SPACE ||
		error_code == CUDA_ERROR_INVALID_PC ||
		error_code == CUDA_ERROR_LAUNCH_FAILED ||
		error_code == CUDA_ERROR_UNKNOWN
		>>
	{
		using category = fatal_error;
	};

	template <CUresult error_code>
	struct error_traits<error_code, std::enable_if_t<
		error_code == CUDA_ERROR_OUT_OF_MEMORY
		>>
	{
		using category = bad_alloc;
	};

	extern template class basic_error<CUDA_ERROR_INVALID_VALUE>;
	extern template class basic_error<CUDA_ERROR_OUT_OF_MEMORY>;
	extern template class basic_error<CUDA_ERROR_NOT_INITIALIZED>;
	extern template class basic_error<CUDA_ERROR_DEINITIALIZED>;
	extern template class basic_error<CUDA_ERROR_PROFILER_DISABLED>;
	extern template class basic_error<CUDA_ERROR_PROFILER_NOT_INITIALIZED>;
	extern template class basic_error<CUDA_ERROR_PROFILER_ALREADY_STARTED>;
	extern template class basic_error<CUDA_ERROR_PROFILER_ALREADY_STOPPED>;
	extern template class basic_error<CUDA_ERROR_NO_DEVICE>;
	extern template class basic_error<CUDA_ERROR_INVALID_DEVICE>;
	extern template class basic_error<CUDA_ERROR_INVALID_IMAGE>;
	extern template class basic_error<CUDA_ERROR_INVALID_CONTEXT>;
	extern template class basic_error<CUDA_ERROR_CONTEXT_ALREADY_CURRENT>;
	extern template class basic_error<CUDA_ERROR_MAP_FAILED>;
	extern template class basic_error<CUDA_ERROR_UNMAP_FAILED>;
	extern template class basic_error<CUDA_ERROR_ARRAY_IS_MAPPED>;
	extern template class basic_error<CUDA_ERROR_ALREADY_MAPPED>;
	extern template class basic_error<CUDA_ERROR_NO_BINARY_FOR_GPU>;
	extern template class basic_error<CUDA_ERROR_ALREADY_ACQUIRED>;
	extern template class basic_error<CUDA_ERROR_NOT_MAPPED>;
	extern template class basic_error<CUDA_ERROR_NOT_MAPPED_AS_ARRAY>;
	extern template class basic_error<CUDA_ERROR_NOT_MAPPED_AS_POINTER>;
	extern template class basic_error<CUDA_ERROR_ECC_UNCORRECTABLE>;
	extern template class basic_error<CUDA_ERROR_UNSUPPORTED_LIMIT>;
	extern template class basic_error<CUDA_ERROR_CONTEXT_ALREADY_IN_USE>;
	extern template class basic_error<CUDA_ERROR_PEER_ACCESS_UNSUPPORTED>;
	extern template class basic_error<CUDA_ERROR_INVALID_PTX>;
	extern template class basic_error<CUDA_ERROR_INVALID_GRAPHICS_CONTEXT>;
	extern template class basic_error<CUDA_ERROR_NVLINK_UNCORRECTABLE>;
	extern template class basic_error<CUDA_ERROR_JIT_COMPILER_NOT_FOUND>;
	extern template class basic_error<CUDA_ERROR_INVALID_SOURCE>;
	extern template class basic_error<CUDA_ERROR_FILE_NOT_FOUND>;
	extern template class basic_error<CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND>;
	extern template class basic_error<CUDA_ERROR_SHARED_OBJECT_INIT_FAILED>;
	extern template class basic_error<CUDA_ERROR_OPERATING_SYSTEM>;
	extern template class basic_error<CUDA_ERROR_INVALID_HANDLE>;
	extern template class basic_error<CUDA_ERROR_NOT_FOUND>;
	extern template class basic_error<CUDA_ERROR_NOT_READY>;
	extern template class basic_error<CUDA_ERROR_ILLEGAL_ADDRESS>;
	extern template class basic_error<CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES>;
	extern template class basic_error<CUDA_ERROR_LAUNCH_TIMEOUT>;
	extern template class basic_error<CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING>;
	extern template class basic_error<CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED>;
	extern template class basic_error<CUDA_ERROR_PEER_ACCESS_NOT_ENABLED>;
	extern template class basic_error<CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE>;
	extern template class basic_error<CUDA_ERROR_CONTEXT_IS_DESTROYED>;
	extern template class basic_error<CUDA_ERROR_ASSERT>;
	extern template class basic_error<CUDA_ERROR_TOO_MANY_PEERS>;
	extern template class basic_error<CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED>;
	extern template class basic_error<CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED>;
	extern template class basic_error<CUDA_ERROR_HARDWARE_STACK_ERROR>;
	extern template class basic_error<CUDA_ERROR_ILLEGAL_INSTRUCTION>;
	extern template class basic_error<CUDA_ERROR_MISALIGNED_ADDRESS>;
	extern template class basic_error<CUDA_ERROR_INVALID_ADDRESS_SPACE>;
	extern template class basic_error<CUDA_ERROR_INVALID_PC>;
	extern template class basic_error<CUDA_ERROR_LAUNCH_FAILED>;
	extern template class basic_error<CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE>;
	extern template class basic_error<CUDA_ERROR_NOT_PERMITTED>;
	extern template class basic_error<CUDA_ERROR_NOT_SUPPORTED>;
	extern template class basic_error<CUDA_ERROR_UNKNOWN>;


	class unknown_error_code : public error
	{
		CUresult error_code;

	public:
		unknown_error_code(CUresult error_code);

		CUresult code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};


	class unexpected_result : public error
	{
		CUresult result;

	public:
		unexpected_result(CUresult result);

		CUresult code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};


	CUresult throw_error(CUresult result);

	inline CUresult succeed(CUresult result)
	{
		if (result != CUDA_SUCCESS)
			throw unknown_error_code(throw_error(result));
		return result;
	}


	template <CUresult expected>
	inline CUresult expect(CUresult result)
	{
		if (result != expected)
			throw unexpected_result(result);
		return result;
	}

	template <CUresult expected_1, CUresult expected_2, CUresult... expected>
	inline CUresult expect(CUresult result)
	{
		if (result != expected_1)
			return expect<expected_2, expected...>(result);
		return result;
	}
}

using CU::throw_error;
using CU::succeed;
using CU::expect;

#endif  // INCLUDED_CUDA_ERROR
