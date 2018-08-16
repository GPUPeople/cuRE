


#ifndef INCLUDED_WIN32_EVENT_HANDLE
#define INCLUDED_WIN32_EVENT_HANDLE

#pragma once

#include "unique_handle.h"
#include "handle.h"


namespace Win32
{
	using unique_hevent = unique_handle<HANDLE, 0, CloseHandleDeleter>;

	unique_hevent createEvent();
}

#endif  // INCLUDED_WIN32_EVENT_HANDLE
