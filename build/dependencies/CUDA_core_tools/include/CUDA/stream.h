


#ifndef INCLUDED_CUDA_STREAM
#define INCLUDED_CUDA_STREAM

#pragma once

#include <cuda.h>

#include <CUDA/unique_handle.h>


namespace CU
{
	struct StreamDestroyDeleter
	{
		void operator ()(CUstream stream) const
		{
			cuStreamDestroy(stream);
		}
	};
	
	using unique_stream = unique_handle<CUstream, nullptr, StreamDestroyDeleter>;
	
	unique_stream createStream(unsigned int flags = CU_STREAM_DEFAULT);
}

#endif  // INCLUDED_CUDA_STREAM
