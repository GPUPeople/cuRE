


#include <CUDA/error.h>
#include <CUDA/event.h>


namespace CU
{
	unique_event createEvent(unsigned int flags)
	{
		CUevent event;
		succeed(cuEventCreate(&event, flags));
		return unique_event(event);
	}
}
