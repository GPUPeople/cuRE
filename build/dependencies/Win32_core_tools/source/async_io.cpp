


#include <win32/error.h>
#include <win32/async_io.h>


namespace Win32
{
	unique_io_completion_port createIOCompletionPort(HANDLE file, ULONG_PTR completion_key, DWORD num_concurrent_threads)
	{
		auto port = CreateIoCompletionPort(file, NULL, completion_key, num_concurrent_threads);

		if(port == NULL)
			throw_last_error();

		return unique_io_completion_port { port };
	}

	unique_io_completion_port createIOCompletionPort(DWORD num_concurrent_threads)
	{
		return createIOCompletionPort(INVALID_HANDLE_VALUE, 0ULL, num_concurrent_threads);
	}

	void associateIOCompletionPort(HANDLE io_completion_port, HANDLE file, ULONG_PTR completion_key)
	{
		auto port = CreateIoCompletionPort(file, io_completion_port, completion_key, 0U);

		if(port != io_completion_port)
			throw_last_error();
	}
}
