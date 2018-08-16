


#include "uniforms.cuh"

extern "C"
{
	__constant__ UniformBuffer camera;
	__constant__ float uniform[48];
	texture<float4, cudaTextureType2D> texf;
	__constant__ uint64_t stippling_mask;
}
