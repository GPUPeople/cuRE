


#ifndef INCLUDED_COM_INIT
#define INCLUDED_COM_INIT

#pragma once

#include <win32/platform.h>
#include <objbase.h>

#include "error.h"


namespace COM
{
	struct scope
	{
	public:
		scope(DWORD init = COINIT_MULTITHREADED)
		{
			succeed(CoInitializeEx(nullptr, init));
		}

		~scope()
		{
			CoUninitialize();
		}
	};
}

#endif  // INCLUDED_COM_INIT
