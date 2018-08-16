


#include <CUDA/error.h>
#include <CUDA/stream.h>


namespace CU
{
	unique_stream createStream(unsigned int flags)
	{
		CUstream stream;
		succeed(cuStreamCreate(&stream, flags));
		return unique_stream(stream);
	}
}
