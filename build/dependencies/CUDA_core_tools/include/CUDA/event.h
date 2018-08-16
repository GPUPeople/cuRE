


#ifndef INCLUDED_CUDA_EVENT
#define INCLUDED_CUDA_EVENT

#pragma once

#include <cuda.h>

#include <CUDA/unique_handle.h>


namespace CU
{
	struct EventDestroyDeleter
	{
		void operator ()(CUevent event) const
		{
			cuEventDestroy(event);
		}
	};
	
	using unique_event = unique_handle<CUevent, nullptr, EventDestroyDeleter>;
	
	unique_event createEvent(unsigned int flags = CU_EVENT_DEFAULT);
}

#endif  // INCLUDED_CUDA_EVENT
