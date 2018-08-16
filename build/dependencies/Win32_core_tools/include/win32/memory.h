


#ifndef INCLUDED_WIN32_MEMORY
#define INCLUDED_WIN32_MEMORY

#pragma once

#include <memory>

#include "platform.h"
#include "unique_handle.h"


namespace Win32
{
	struct HeapDestroyDeleter
	{
		void operator ()(HANDLE heap) const
		{
			HeapDestroy(heap);
		}
	};

	using unique_heap = unique_handle<HANDLE, 0, HeapDestroyDeleter>;

	unique_heap createHeap(DWORD options, SIZE_T initial_size, SIZE_T maximum_size = 0U);


	class HeapFreeDeleter
	{
		HANDLE heap;

	public:
		HeapFreeDeleter(HANDLE heap)
			: heap(heap)
		{
		}

		void operator ()(void* p) const
		{
			HeapFree(heap, 0U, p);
		}
	};

	struct DefaultHeapDeleter
	{
		void operator ()(void* p) const
		{
			HeapFree(GetProcessHeap(), 0U, p);
		}
	};

	template <typename T, typename Del = DefaultHeapDeleter>
	using unique_heap_ptr = std::unique_ptr<T, Del>;


	struct GlobalFreeDeleter
	{
		void operator ()(HGLOBAL hmem) const
		{
			GlobalFree(hmem);
		}
	};

	using unique_hglobal = unique_handle<HGLOBAL, 0, GlobalFreeDeleter>;


	struct LocalFreeDeleter
	{
		void operator ()(HLOCAL hmem) const
		{
			LocalFree(hmem);
		}
	};

	using unique_hlocal = unique_handle<HLOCAL, 0, LocalFreeDeleter>;
}

#endif  // INCLUDED_WIN32_MODULE_HANDLE
