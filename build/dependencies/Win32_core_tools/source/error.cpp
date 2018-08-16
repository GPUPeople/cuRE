


#include <win32/memory.h>
#include <win32/unicode.h>

#include <win32/error.h>


namespace Win32
{
	std::wstring formatErrorMessage(DWORD error_code)
	{
		WCHAR* buffer;
		auto length = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, error_code, 0, reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);
		if (length == 0)
			throw_last_error();
		Win32::unique_heap_ptr<WCHAR> msg(buffer);
		return { msg.get(), length };
	}


	error::error(DWORD error_code)
		: error_code(error_code)
	{
	}

	const char* error::what() const noexcept
	{
		return "Win32 error";
	}

	std::string error::message() const
	{
		return narrow(formatErrorMessage(error_code));
	}


	void throw_last_error()
	{
		auto err = GetLastError();

		if (FAILED(HRESULT_FROM_WIN32(err)))
			throw error(err);
	}
}
