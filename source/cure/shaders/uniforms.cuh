


#ifndef CAMERA_UNIFORM_INCLUDED
#define CAMERA_UNIFORM_INCLUDED

#include <math/vector.h>
#include <math/matrix.h>
#include <cstdint>


struct UniformBuffer
{
	alignas(64) math::affine_float4x4 V;
	alignas(64) math::affine_float4x4 V_inv;
	alignas(64) math::float4x4 P;
	alignas(64) math::float4x4 P_inv;
	alignas(64) math::float4x4 PV;
	alignas(64) math::float4x4 PV_inv;
	alignas(64) math::float3 position;
	alignas(64) math::float4x4 PVM;
	alignas(64) math::float4x4 PVM_inv;
};

constexpr float CHECKERBOARD_RADIUS = 256.f;
constexpr float CHECKERBOARD_EPSILON = 0.022f;

extern "C"
{
	extern __constant__ UniformBuffer camera;
	extern __constant__ float uniform[48];
	extern texture<float4, cudaTextureType2D> texf;
	extern __constant__ uint64_t stippling_mask;
}

//#define CHECKER

#endif
