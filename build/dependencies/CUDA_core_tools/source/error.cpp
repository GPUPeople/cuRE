


#include <CUDA/error.h>


namespace
{
	const char* getErrorName(CUresult error_code) noexcept
	{
		const char* string;
		if (cuGetErrorName(error_code, &string) != CUDA_SUCCESS)
			return "invalid error code";
		return string;
	}

	const char* getErrorString(CUresult error_code) noexcept
	{
		const char* string;
		if (cuGetErrorString(error_code, &string) != CUDA_SUCCESS)
			return "invalid error code";
		return string;
	}
}

namespace CU
{
	template <CUresult error_code>
	CUresult basic_error<error_code>::code() const noexcept
	{
		return error_code;
	}

	template <CUresult error_code>
	const char* basic_error<error_code>::name() const noexcept
	{
		return getErrorName(error_code);
	}

	template <CUresult error_code>
	const char* basic_error<error_code>::what() const noexcept
	{
		return getErrorString(error_code);
	}

	template class basic_error<CUDA_ERROR_INVALID_VALUE>;
	template class basic_error<CUDA_ERROR_OUT_OF_MEMORY>;
	template class basic_error<CUDA_ERROR_NOT_INITIALIZED>;
	template class basic_error<CUDA_ERROR_DEINITIALIZED>;
	template class basic_error<CUDA_ERROR_PROFILER_DISABLED>;
	template class basic_error<CUDA_ERROR_PROFILER_NOT_INITIALIZED>;
	template class basic_error<CUDA_ERROR_PROFILER_ALREADY_STARTED>;
	template class basic_error<CUDA_ERROR_PROFILER_ALREADY_STOPPED>;
	template class basic_error<CUDA_ERROR_NO_DEVICE>;
	template class basic_error<CUDA_ERROR_INVALID_DEVICE>;
	template class basic_error<CUDA_ERROR_INVALID_IMAGE>;
	template class basic_error<CUDA_ERROR_INVALID_CONTEXT>;
	template class basic_error<CUDA_ERROR_CONTEXT_ALREADY_CURRENT>;
	template class basic_error<CUDA_ERROR_MAP_FAILED>;
	template class basic_error<CUDA_ERROR_UNMAP_FAILED>;
	template class basic_error<CUDA_ERROR_ARRAY_IS_MAPPED>;
	template class basic_error<CUDA_ERROR_ALREADY_MAPPED>;
	template class basic_error<CUDA_ERROR_NO_BINARY_FOR_GPU>;
	template class basic_error<CUDA_ERROR_ALREADY_ACQUIRED>;
	template class basic_error<CUDA_ERROR_NOT_MAPPED>;
	template class basic_error<CUDA_ERROR_NOT_MAPPED_AS_ARRAY>;
	template class basic_error<CUDA_ERROR_NOT_MAPPED_AS_POINTER>;
	template class basic_error<CUDA_ERROR_ECC_UNCORRECTABLE>;
	template class basic_error<CUDA_ERROR_UNSUPPORTED_LIMIT>;
	template class basic_error<CUDA_ERROR_CONTEXT_ALREADY_IN_USE>;
	template class basic_error<CUDA_ERROR_PEER_ACCESS_UNSUPPORTED>;
	template class basic_error<CUDA_ERROR_INVALID_PTX>;
	template class basic_error<CUDA_ERROR_INVALID_GRAPHICS_CONTEXT>;
	template class basic_error<CUDA_ERROR_NVLINK_UNCORRECTABLE>;
	template class basic_error<CUDA_ERROR_JIT_COMPILER_NOT_FOUND>;
	template class basic_error<CUDA_ERROR_INVALID_SOURCE>;
	template class basic_error<CUDA_ERROR_FILE_NOT_FOUND>;
	template class basic_error<CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND>;
	template class basic_error<CUDA_ERROR_SHARED_OBJECT_INIT_FAILED>;
	template class basic_error<CUDA_ERROR_OPERATING_SYSTEM>;
	template class basic_error<CUDA_ERROR_INVALID_HANDLE>;
	template class basic_error<CUDA_ERROR_NOT_FOUND>;
	template class basic_error<CUDA_ERROR_NOT_READY>;
	template class basic_error<CUDA_ERROR_ILLEGAL_ADDRESS>;
	template class basic_error<CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES>;
	template class basic_error<CUDA_ERROR_LAUNCH_TIMEOUT>;
	template class basic_error<CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING>;
	template class basic_error<CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED>;
	template class basic_error<CUDA_ERROR_PEER_ACCESS_NOT_ENABLED>;
	template class basic_error<CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE>;
	template class basic_error<CUDA_ERROR_CONTEXT_IS_DESTROYED>;
	template class basic_error<CUDA_ERROR_ASSERT>;
	template class basic_error<CUDA_ERROR_TOO_MANY_PEERS>;
	template class basic_error<CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED>;
	template class basic_error<CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED>;
	template class basic_error<CUDA_ERROR_HARDWARE_STACK_ERROR>;
	template class basic_error<CUDA_ERROR_ILLEGAL_INSTRUCTION>;
	template class basic_error<CUDA_ERROR_MISALIGNED_ADDRESS>;
	template class basic_error<CUDA_ERROR_INVALID_ADDRESS_SPACE>;
	template class basic_error<CUDA_ERROR_INVALID_PC>;
	template class basic_error<CUDA_ERROR_LAUNCH_FAILED>;
	template class basic_error<CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE>;
	template class basic_error<CUDA_ERROR_NOT_PERMITTED>;
	template class basic_error<CUDA_ERROR_NOT_SUPPORTED>;
	template class basic_error<CUDA_ERROR_UNKNOWN>;


	unknown_error_code::unknown_error_code(CUresult error_code)
		: error_code(error_code)
	{
	}

	CUresult unknown_error_code::code() const noexcept
	{
		return error_code;
	}

	const char* unknown_error_code::name() const noexcept
	{
		return getErrorName(error_code);
	}
	
	const char* unknown_error_code::what() const noexcept
	{
		return getErrorString(error_code);
	}


	unexpected_result::unexpected_result(CUresult result)
		: result(result)
	{
	}

	CUresult unexpected_result::code() const noexcept
	{
		return result;
	}

	const char* unexpected_result::name() const noexcept
	{
		return getErrorName(result);
	}

	const char* unexpected_result::what() const noexcept
	{
		return getErrorString(result);
	}


	CUresult throw_error(CUresult result)
	{
		switch (result)
		{
		case CUDA_SUCCESS:
			break;
		case CUDA_ERROR_INVALID_VALUE:
			throw basic_error<CUDA_ERROR_INVALID_VALUE>();
		case CUDA_ERROR_OUT_OF_MEMORY:
			throw basic_error<CUDA_ERROR_OUT_OF_MEMORY>();
		case CUDA_ERROR_NOT_INITIALIZED:
			throw basic_error<CUDA_ERROR_NOT_INITIALIZED>();
		case CUDA_ERROR_DEINITIALIZED:
			throw basic_error<CUDA_ERROR_DEINITIALIZED>();
		case CUDA_ERROR_PROFILER_DISABLED:
			throw basic_error<CUDA_ERROR_PROFILER_DISABLED>();
		case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
			throw basic_error<CUDA_ERROR_PROFILER_NOT_INITIALIZED>();
		case CUDA_ERROR_PROFILER_ALREADY_STARTED:
			throw basic_error<CUDA_ERROR_PROFILER_ALREADY_STARTED>();
		case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
			throw basic_error<CUDA_ERROR_PROFILER_ALREADY_STOPPED>();
		case CUDA_ERROR_NO_DEVICE:
			throw basic_error<CUDA_ERROR_NO_DEVICE>();
		case CUDA_ERROR_INVALID_DEVICE:
			throw basic_error<CUDA_ERROR_INVALID_DEVICE>();
		case CUDA_ERROR_INVALID_IMAGE:
			throw basic_error<CUDA_ERROR_INVALID_IMAGE>();
		case CUDA_ERROR_INVALID_CONTEXT:
			throw basic_error<CUDA_ERROR_INVALID_CONTEXT>();
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
			throw basic_error<CUDA_ERROR_CONTEXT_ALREADY_CURRENT>();
		case CUDA_ERROR_MAP_FAILED:
			throw basic_error<CUDA_ERROR_MAP_FAILED>();
		case CUDA_ERROR_UNMAP_FAILED:
			throw basic_error<CUDA_ERROR_UNMAP_FAILED>();
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			throw basic_error<CUDA_ERROR_ARRAY_IS_MAPPED>();
		case CUDA_ERROR_ALREADY_MAPPED:
			throw basic_error<CUDA_ERROR_ALREADY_MAPPED>();
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			throw basic_error<CUDA_ERROR_NO_BINARY_FOR_GPU>();
		case CUDA_ERROR_ALREADY_ACQUIRED:
			throw basic_error<CUDA_ERROR_ALREADY_ACQUIRED>();
		case CUDA_ERROR_NOT_MAPPED:
			throw basic_error<CUDA_ERROR_NOT_MAPPED>();
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			throw basic_error<CUDA_ERROR_NOT_MAPPED_AS_ARRAY>();
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			throw basic_error<CUDA_ERROR_NOT_MAPPED_AS_POINTER>();
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			throw basic_error<CUDA_ERROR_ECC_UNCORRECTABLE>();
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			throw basic_error<CUDA_ERROR_UNSUPPORTED_LIMIT>();
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
			throw basic_error<CUDA_ERROR_CONTEXT_ALREADY_IN_USE>();
		case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
			throw basic_error<CUDA_ERROR_PEER_ACCESS_UNSUPPORTED>();
		case CUDA_ERROR_INVALID_PTX:
			throw basic_error<CUDA_ERROR_INVALID_PTX>();
		case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
			throw basic_error<CUDA_ERROR_INVALID_GRAPHICS_CONTEXT>();
		case CUDA_ERROR_NVLINK_UNCORRECTABLE:
			throw basic_error<CUDA_ERROR_NVLINK_UNCORRECTABLE>();
		case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
			throw basic_error<CUDA_ERROR_JIT_COMPILER_NOT_FOUND>();
		case CUDA_ERROR_INVALID_SOURCE:
			throw basic_error<CUDA_ERROR_INVALID_SOURCE>();
		case CUDA_ERROR_FILE_NOT_FOUND:
			throw basic_error<CUDA_ERROR_FILE_NOT_FOUND>();
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			throw basic_error<CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND>();
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			throw basic_error<CUDA_ERROR_SHARED_OBJECT_INIT_FAILED>();
		case CUDA_ERROR_OPERATING_SYSTEM:
			throw basic_error<CUDA_ERROR_OPERATING_SYSTEM>();
		case CUDA_ERROR_INVALID_HANDLE:
			throw basic_error<CUDA_ERROR_INVALID_HANDLE>();
		case CUDA_ERROR_NOT_FOUND:
			throw basic_error<CUDA_ERROR_NOT_FOUND>();
		case CUDA_ERROR_NOT_READY:
			throw basic_error<CUDA_ERROR_NOT_READY>();
		case CUDA_ERROR_ILLEGAL_ADDRESS:
			throw basic_error<CUDA_ERROR_ILLEGAL_ADDRESS>();
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			throw basic_error<CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES>();
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			throw basic_error<CUDA_ERROR_LAUNCH_TIMEOUT>();
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			throw basic_error<CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING>();
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
			throw basic_error<CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED>();
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
			throw basic_error<CUDA_ERROR_PEER_ACCESS_NOT_ENABLED>();
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
			throw basic_error<CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE>();
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:
			throw basic_error<CUDA_ERROR_CONTEXT_IS_DESTROYED>();
		case CUDA_ERROR_ASSERT:
			throw basic_error<CUDA_ERROR_ASSERT>();
		case CUDA_ERROR_TOO_MANY_PEERS:
			throw basic_error<CUDA_ERROR_TOO_MANY_PEERS>();
		case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
			throw basic_error<CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED>();
		case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
			throw basic_error<CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED>();
		case CUDA_ERROR_HARDWARE_STACK_ERROR:
			throw basic_error<CUDA_ERROR_HARDWARE_STACK_ERROR>();
		case CUDA_ERROR_ILLEGAL_INSTRUCTION:
			throw basic_error<CUDA_ERROR_ILLEGAL_INSTRUCTION>();
		case CUDA_ERROR_MISALIGNED_ADDRESS:
			throw basic_error<CUDA_ERROR_MISALIGNED_ADDRESS>();
		case CUDA_ERROR_INVALID_ADDRESS_SPACE:
			throw basic_error<CUDA_ERROR_INVALID_ADDRESS_SPACE>();
		case CUDA_ERROR_INVALID_PC:
			throw basic_error<CUDA_ERROR_INVALID_PC>();
		case CUDA_ERROR_LAUNCH_FAILED:
			throw basic_error<CUDA_ERROR_LAUNCH_FAILED>();
		case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
			throw basic_error<CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE>();
		case CUDA_ERROR_NOT_PERMITTED:
			throw basic_error<CUDA_ERROR_NOT_PERMITTED>();
		case CUDA_ERROR_NOT_SUPPORTED:
			throw basic_error<CUDA_ERROR_NOT_SUPPORTED>();
		case CUDA_ERROR_UNKNOWN:
			throw basic_error<CUDA_ERROR_UNKNOWN>();
		}
		return result;
	}
}
