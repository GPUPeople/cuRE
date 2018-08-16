


#ifndef INCLUDED_WIN32_FILE
#define INCLUDED_WIN32_FILE

#pragma once

#include <utility>
#include <string>

#include "platform.h"
#include "error.h"
#include "handle.h"
#include "unique_handle.h"


namespace Win32
{
	using unique_hfile = unique_handle<HANDLE, INVALID_HANDLE_VALUE, CloseHandleDeleter>;

	unique_hfile createFile(const wchar_t* file_name, DWORD access, DWORD share_mode, DWORD create, DWORD attributes);
	unique_hfile createFile(const char* file_name, DWORD access, DWORD share_mode, DWORD create, DWORD attributes);

	void read(HANDLE file, char* buffer, size_t size);
	void write(HANDLE file, const char* buffer, size_t size);
}

#endif  // INCLUDED_WIN32_FILE
