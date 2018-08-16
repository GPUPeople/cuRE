


#ifndef INCLUDED_CAMERA
#define INCLUDED_CAMERA

#pragma once

#include <interface.h>

#include <math/matrix.h>
//#include <cure/shaders/camera.cuh>

struct INTERFACE Camera
{
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

	virtual void writeUniformBuffer(UniformBuffer* params, float aspect) const = 0;

protected:
	Camera() = default;
	Camera(const Camera&) = default;
	Camera& operator =(const Camera&) = default;
	~Camera() = default;
};

#endif  // INCLUDED_CAMERA
