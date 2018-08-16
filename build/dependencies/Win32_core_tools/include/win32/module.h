


#ifndef INCLUDED_WIN32_MODULE
#define INCLUDED_WIN32_MODULE

#pragma once

#include "platform.h"
#include "unique_handle.h"


namespace Win32
{
	struct FreeLibraryDeleter
	{
		void operator ()(HMODULE module) const
		{
			FreeLibrary(module);
		}
	};

	using unique_hmodule = unique_handle<HMODULE, 0, FreeLibraryDeleter>;

	unique_hmodule loadLibrary(const WCHAR* filename);
	unique_hmodule loadLibrary(const WCHAR* filename, HANDLE file, DWORD flags);
	unique_hmodule loadLibrary(const CHAR* filename);
	unique_hmodule loadLibrary(const CHAR* filename, HANDLE file, DWORD flags);
}

#endif  // INCLUDED_WIN32_MODULE
