


#ifndef INCLUDED_CUDA_CONTEXT
#define INCLUDED_CUDA_CONTEXT

#pragma once

#include <cuda.h>

#include <CUDA/unique_handle.h>

#include "error.h"


namespace CU
{
	struct CtxDestroyDeleter
	{
		void operator ()(CUcontext context) const
		{
			cuCtxDestroy(context);
		}
	};
	
	using unique_context = unique_handle<CUcontext, nullptr, CtxDestroyDeleter>;
	
	inline CUcontext getCurrentContext()
	{
		CUcontext ctx;
		succeed(cuCtxGetCurrent(&ctx));
		return ctx;
	}
	
	struct context_scope
	{
		context_scope(CUcontext ctx)
		{
			succeed(cuCtxPushCurrent(ctx));
		}
		
		~context_scope()
		{
			CUcontext old_ctx;
			succeed(cuCtxPopCurrent(&old_ctx));
		}
	};
	
	unique_context createContext(unsigned int flags = 0U, CUdevice device = 0);
}

#endif  // INCLUDED_CUDA_CONTEXT
