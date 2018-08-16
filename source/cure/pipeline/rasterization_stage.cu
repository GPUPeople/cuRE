


#include <stdio.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "triangle_buffer.cuh"
#include "index_queue.cuh"
#include "config.h"

#include "viewport.cuh"


#define RASTERIZATION_STAGE_GLOBAL
#include "rasterization_stage.cuh"


extern "C"
{
	__constant__ Viewport viewport;
	__constant__ float4 pixel_scale;

	__global__
	void initRasterizationStage()
	{
		virtual_rasterizers.init();
		triangle_buffer.init();
		rasterizer_queue.init();
	}

	extern __device__ int geometryProducingBlocksCount;

	__global__
	void prepareRasterizationNewPrimitive(unsigned int rasterizer_count)
	//void prepareRasterizationNewPrimitive()
	{
		virtual_rasterizers.init();
		if (blockIdx.x == 0 && threadIdx.x == 0)
			geometryProducingBlocksCount = rasterizer_count;

		if (ENFORCE_PRIMITIVE_ORDER)
			rasterizer_queue.newPrimitive();
	}

	__global__
	void checkQueues()
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		int count;
		unsigned int f, b;
		for (int i = id; i < MAX_NUM_RASTERIZERS; i += gridDim.x * blockDim.x)
		{ 
			rasterizer_queue.index_queue.readState(i, count, f, b);
			if (count != 0)
				printf("raster queue %d not empty: %d with f %d b %d\n", i, count, f, b);
		}
		
		triangle_buffer.check();
	}
}
