


#include "error.h"


namespace NVRTC
{
	error::error(nvrtcResult error_code)
		: error_code(error_code)
	{
	}

	nvrtcResult error::code() const noexcept
	{
		return error_code;
	}

	const char* error::what() const noexcept
	{
		return nvrtcGetErrorString(error_code);
	}


	unknown_error_code::unknown_error_code(nvrtcResult error_code)
		: error(error_code)
	{
	}


	unexpected_result::unexpected_result(nvrtcResult result)
		: error(result)
	{
	}


	nvrtcResult throw_error(nvrtcResult result)
	{
		switch (result)
		{
		case NVRTC_ERROR_OUT_OF_MEMORY:
		case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
		case NVRTC_ERROR_INVALID_INPUT:
		case NVRTC_ERROR_INVALID_PROGRAM:
		case NVRTC_ERROR_INVALID_OPTION:
		case NVRTC_ERROR_COMPILATION:
		case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
		case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
		case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
		case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
		case NVRTC_ERROR_INTERNAL_ERROR:
			throw error(result);
		}

		return result;
	}
}
