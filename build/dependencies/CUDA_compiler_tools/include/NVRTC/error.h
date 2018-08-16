


#ifndef INCLUDED_NVRTC_ERROR
#define INCLUDED_NVRTC_ERROR

#pragma once

#include <exception>

#include <nvrtc.h>


namespace NVRTC
{
	class error : public std::exception
	{
		nvrtcResult error_code;

	public:
		error(nvrtcResult error_code);

		virtual nvrtcResult code() const noexcept;
		const char* what() const noexcept override;
	};

	class unknown_error_code : public error
	{
	public:
		unknown_error_code(nvrtcResult error_code);
	};

	class unexpected_result : public error
	{
	public:
		unexpected_result(nvrtcResult result);
	};



	nvrtcResult throw_error(nvrtcResult result);

	inline nvrtcResult succeed(nvrtcResult result)
	{
		if (result != NVRTC_SUCCESS)
			throw unknown_error_code(throw_error(result));
		return result;
	}


	template <nvrtcResult expected>
	inline nvrtcResult expect(nvrtcResult result)
	{
		if (result != expected)
			throw unexpected_result(result);
		return result;
	}

	template <nvrtcResult expected_1, nvrtcResult expected_2, nvrtcResult... expected>
	inline nvrtcResult expect(nvrtcResult result)
	{
		if (result != expected_1)
			return expect<expected_2, expected...>(result);
		return result;
	}
}

using NVRTC::throw_error;
using NVRTC::succeed;
using NVRTC::expect;

#endif  // INCLUDED_NVRTC_ERROR
