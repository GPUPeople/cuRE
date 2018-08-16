


#include <win32/error.h>
#include <win32/unicode.h>

#include <COM/error.h>


namespace COM
{
	error::error(HRESULT hresult)
		: hresult(hresult)
	{
	}

	const char* error::what() const noexcept
	{
		return "COM error";
	}

	std::string error::message() const
	{
		return narrow(Win32::formatErrorMessage(hresult));
	}
}
