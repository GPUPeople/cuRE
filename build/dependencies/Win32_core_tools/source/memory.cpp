


#include <win32/error.h>

#include <win32/memory.h>


namespace Win32
{
	unique_heap createHeap(DWORD options, SIZE_T initial_size, SIZE_T maximum_size)
	{
		HANDLE heap = HeapCreate(options, initial_size, maximum_size);
		if (heap == 0)
			throw_last_error();
		return unique_heap { heap };
	}
}
