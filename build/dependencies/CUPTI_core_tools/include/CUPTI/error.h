


#ifndef INCLUDED_CUPTI_ERROR
#define INCLUDED_CUPTI_ERROR

#pragma once

#include <exception>

#include <cupti.h>


namespace CUPTI
{
	class error : public std::exception
	{
		CUptiResult result;

	public:
		error(CUptiResult result)
		  : result(result)
		{
		}

		const char* what() const noexcept
		{
			return "CUPTI fail";
		}
	};

	inline CUptiResult succeed(CUptiResult result)
	{
		if (result != CUPTI_SUCCESS)
			throw CUPTI::error(result);
		return result;
	}
}

using CUPTI::succeed;

#endif  // INCLUDED_CUPTI_ERROR
