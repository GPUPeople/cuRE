


#include "Noncamera.h"


namespace
{
	math::float4x4 proj(float fov, float aspect, float nearz, float farz)
	{
		return  math::float4x4( 1.0f,  0.0f,  0.0f,  0.0f,
			                     0.0f,  1.0f,  0.0f,  0.0f,
			                     0.0f,  0.0f, -1.0f,  0.0f,
			                     0.0f,  0.0f, -1.0f,  0.0f);
	}
}

void Noncamera::attach(const Navigator* navigator)
{
}

void Noncamera::writeUniformBuffer(UniformBuffer* buffer, float aspect) const
{
	buffer->V = math::identity<math::affine_float4x4>();
	buffer->V_inv = math::identity<math::affine_float4x4>();
	buffer->position = math::float3(0.0f, 0.0f, 0.0f);

	buffer->P = math::identity<math::float4x4>();
	buffer->P_inv = math::identity<math::float4x4>();
	buffer->PV = math::identity<math::float4x4>();
	buffer->PV_inv = math::identity<math::float4x4>();
}
