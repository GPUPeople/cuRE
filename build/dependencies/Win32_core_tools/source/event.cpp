


#include <win32/error.h>

#include <win32/event.h>


namespace Win32
{
	unique_hevent createEvent()
	{
		HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		if (h == 0)
			throw_last_error();
		return unique_hevent { h };
	}
}
