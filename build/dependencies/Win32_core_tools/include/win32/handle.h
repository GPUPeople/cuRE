


#ifndef INCLUDED_WIN32_HANDLE
#define INCLUDED_WIN32_HANDLE

#pragma once

#include "platform.h"


namespace Win32
{
	struct CloseHandleDeleter
	{
		void operator ()(HANDLE h) const
		{
			CloseHandle(h);
		}
	};

	HANDLE duplicateHandle(HANDLE source_process, HANDLE source_handle, HANDLE target_process, DWORD desired_access, BOOL inheritable, DWORD options = 0U);

	DWORD getHandleInformation(HANDLE h);
	void setHandleInformation(HANDLE h, DWORD mask, DWORD flags);
}

#endif  // INCLUDED_WIN32_HANDLE
