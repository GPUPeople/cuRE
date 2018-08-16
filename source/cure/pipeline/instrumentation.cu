


#define INSTRUMENTATION_GLOBAL
#include "instrumentation.cuh"

extern "C"
{
	__global__
	void initInstrumentation()
	{
		instrumentation.reset();
	}

	__global__
	void resetInstrumentation()
	{
		instrumentation.reset();
	}

	__global__
	void readInstrumentationData(Instrumentation::TimingResult<unsigned int>::data_t* per_block_timing_buffer, unsigned int buffer_stride, unsigned int num_blocks)
	{
		instrumentation.read(per_block_timing_buffer, buffer_stride, num_blocks);
	}
}
