


#ifndef INCLUDED_COM_ERROR
#define INCLUDED_COM_ERROR

#pragma once

#include <exception>
#include <string>

#include <win32/platform.h>


namespace COM
{
	class error : public std::exception
	{
		HRESULT hresult;

	public:
		explicit error(HRESULT hresult);

		const char* what() const noexcept override;

		std::string message() const;
	};


	inline HRESULT throw_error(HRESULT hresult)
	{
		if (FAILED(hresult))
			throw error(hresult);
		return hresult;
	}

	inline HRESULT succeed(HRESULT hresult)
	{
		return throw_error(hresult);
	}
}

#endif  // INCLUDED_COM_ERROR
