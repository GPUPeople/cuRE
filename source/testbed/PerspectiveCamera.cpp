


#include "PerspectiveCamera.h"

namespace
{
	math::float4x4 proj(float fov, float aspect, float nearz, float farz)
	{
		const float s2 = 1.0f / std::tan(fov * 0.5f);
		const float s1 = s2 / aspect;
		const float z1 = (farz + nearz) / (farz - nearz);
		const float z2 = -2.0f * nearz * farz / (farz - nearz);

		return math::float4x4(s1, 0.0f, 0.0f, 0.0f,
		                      0.0f, s2, 0.0f, 0.0f,
		                      0.0f, 0.0f, z1, z2,
		                      0.0f, 0.0f, 1.0f, 0.0f);
	}
}

PerspectiveCamera::PerspectiveCamera(float fov, float z_near, float z_far)
	: fov(fov),
	  nearz(z_near),
	  farz(z_far),
	  navigator(nullptr)
{
}

void PerspectiveCamera::attach(const Navigator* navigator)
{
	PerspectiveCamera::navigator = navigator;
}

void PerspectiveCamera::writeUniformBuffer(UniformBuffer* buffer, float aspect) const
{
	if (navigator)
	{
		navigator->writeWorldToLocalTransform(&buffer->V);
		navigator->writeLocalToWorldTransform(&buffer->V_inv);
		navigator->writePosition(&buffer->position);
	}
	else
	{
		buffer->V = math::identity<math::affine_float4x4>();
		buffer->V_inv = math::identity<math::affine_float4x4>();
		buffer->position = math::float3(0.0f, 0.0f, 0.0f);
	}

	buffer->P = proj(fov, aspect, nearz, farz);
	buffer->P_inv = inverse(buffer->P);
	buffer->PV = buffer->P * buffer->V;
	buffer->PV_inv = buffer->V_inv * buffer->P_inv;
}
