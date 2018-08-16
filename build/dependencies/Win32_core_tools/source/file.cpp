


#include <win32/file.h>


namespace Win32
{
	unique_hfile createFile(const wchar_t* file_name, DWORD access, DWORD share_mode, DWORD create, DWORD attributes)
	{
		HANDLE h = CreateFileW(file_name, access, share_mode, nullptr, create, attributes, 0);
		if (h == INVALID_HANDLE_VALUE)
			throw_last_error();
		return unique_hfile { h };
	}

	unique_hfile createFile(const char* file_name, DWORD access, DWORD share_mode, DWORD create, DWORD attributes)
	{
		HANDLE h = CreateFileA(file_name, access, share_mode, nullptr, create, attributes, 0);
		if (h == INVALID_HANDLE_VALUE)
			throw_last_error();
		return unique_hfile { h };
	}

	void read(HANDLE file, char* buffer, size_t size)
	{
		DWORD bytes_read;
		if (ReadFile(file, buffer, static_cast<DWORD>(size), &bytes_read, nullptr) != TRUE || bytes_read != size)
			throw_last_error();
	}

	void write(HANDLE file, const char* buffer, size_t size)
	{
		DWORD bytes_written;
		if (WriteFile(file, buffer, static_cast<DWORD>(size), &bytes_written, nullptr) != TRUE || bytes_written != size)
			throw_last_error();
	}
}
