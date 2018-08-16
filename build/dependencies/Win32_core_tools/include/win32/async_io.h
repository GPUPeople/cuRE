


#ifndef INCLUDED_WIN32_ASYNC_IO
#define INCLUDED_WIN32_ASYNC_IO

#pragma once

#include "platform.h"
#include "handle.h"
#include "unique_handle.h"


namespace Win32
{
	using unique_io_completion_port = unique_handle<HANDLE, INVALID_HANDLE_VALUE, CloseHandleDeleter>;

	unique_io_completion_port createIOCompletionPort(DWORD num_concurrent_threads = 0U);
	unique_io_completion_port createIOCompletionPort(HANDLE file, ULONG_PTR completion_key, DWORD num_concurrent_threads = 0U);

	void associateIOCompletionPort(HANDLE io_completion_port, HANDLE file, ULONG_PTR completion_key);
}

#endif  // INCLUDED_WIN32_ASYNC_IO
