


#include <win32/error.h>

#include <win32/handle.h>


namespace Win32
{
	HANDLE duplicateHandle(HANDLE source_process, HANDLE source_handle, HANDLE target_process, DWORD desired_access, BOOL inheritable, DWORD options)
	{
		HANDLE handle;
		if (!DuplicateHandle(source_process, source_handle, target_process, &handle, desired_access, inheritable, options))
			throw_last_error();
		return handle;
	}

	DWORD getHandleInformation(HANDLE h)
	{
		DWORD flags;
		if (!GetHandleInformation(h, &flags))
			throw_last_error();
		return flags;
	}

	void setHandleInformation(HANDLE h, DWORD mask, DWORD flags)
	{
		if (!SetHandleInformation(h, mask, flags))
			throw_last_error();
	}
}
