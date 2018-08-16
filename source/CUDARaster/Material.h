


#ifndef INCLUDED_CUDARASTER_MATERIAL
#define INCLUDED_CUDARASTER_MATERIAL

#pragma once

#include <interface.h>
#include <Resource.h>
#include <math/matrix.h>


namespace CUDARaster
{
	class INTERFACE Material : public ::Material
	{
	public:
		virtual void setModel(const math::float3x4& mat) = 0;
		virtual void setCamera(const math::float4x4& PV, const math::float3& pos) = 0;
		virtual void remove() = 0;
	};
}

#endif // INCLUDED_CURE_SHADERS_HOST_SHADER_INTERFACE
