


#ifndef INCLUDED_WIN32_ERROR
#define INCLUDED_WIN32_ERROR

#pragma once

#include <exception>
#include <string>

#include "platform.h"


namespace Win32
{
	class error : public std::exception
	{
		DWORD error_code;

	public:
		explicit error(DWORD error_code);

		const char* what() const noexcept override;

		std::string message() const;
	};

	void throw_last_error();


	std::wstring formatErrorMessage(DWORD error_code);
}

#endif  // INCLUDED_WIN32_ERROR
