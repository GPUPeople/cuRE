


#include "fragment_processing.cuh"
#include "geometry_processing.cuh"
#include "shaders/clipspace.cuh"
#include "DepthStorage.cuh"

extern "C" 
{
	__global__ void runFragmentStageClipSpace(unsigned int num_triangles)
	{
		using namespace FreePipe;

		unsigned int triangle_id = blockDim.x * blockIdx.x + threadIdx.x;
		if(triangle_id < num_triangles)
			process_fragments_h<FreePipe::Shaders::ClipSpaceVertexShader, FreePipe::Shaders::ClipSpaceFragmentShader>(c_depthBuffer, c_bufferDims[0], c_bufferDims[1], pixel_step[0], pixel_step[1], triangle_id);
	}
}
