


#include <CUDA/error.h>
#include <CUDA/context.h>


namespace CU
{
	unique_context createContext(unsigned int flags, CUdevice device)
	{
		CUcontext context;
		succeed(cuCtxCreate(&context, flags, device));
		return unique_context(context);
	}
}
