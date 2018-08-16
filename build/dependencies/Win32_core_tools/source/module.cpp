


#include <win32/error.h>
#include <win32/module.h>


namespace Win32
{
	unique_hmodule loadLibrary(const WCHAR* filename)
	{
		HMODULE module = LoadLibraryW(filename);
		if (module == 0)
			throw_last_error();
		return unique_hmodule { module };
	}

	unique_hmodule loadLibrary(const WCHAR* filename, HANDLE file, DWORD flags)
	{
		HMODULE module = LoadLibraryExW(filename, file, flags);
		if (module == 0)
			throw_last_error();
		return unique_hmodule { module };
	}

	unique_hmodule loadLibrary(const CHAR* filename)
	{
		HMODULE module = LoadLibraryA(filename);
		if (module == 0)
			throw_last_error();
		return unique_hmodule { module };
	}

	unique_hmodule loadLibrary(const CHAR* filename, HANDLE file, DWORD flags)
	{
		HMODULE module = LoadLibraryExA(filename, file, flags);
		if (module == 0)
			throw_last_error();
		return unique_hmodule { module };
	}
}
