


#ifndef INCLUDED_CUDARASTER_MATERIAL_EYECANDY
#define INCLUDED_CUDARASTER_MATERIAL_EYECANDY

#include <CUDA/module.h>
#include "../Material.h"
#include "../Shaders.hpp"

namespace CUDARaster
{
	class EyeCandyMaterial : public CUDARaster::Material
	{
		EyeCandyMaterial(const EyeCandyMaterial&) = delete;
		EyeCandyMaterial& operator =(const EyeCandyMaterial&) = delete;

		CUdeviceptr phongData;
		CUmodule module;
		CUfunction kernel_fragment_processing;

		CUdeviceptr matData;
		FW::Material matbase;

	public:
		EyeCandyMaterial(CUmodule module);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;

		void setModel(const math::float3x4& mat);
		void setCamera(const math::float4x4& PV, const math::float3& pos);
		void remove()
		{
		}
	};
}

#endif  // INCLUDED_CUDARASTER_MATERIAL_EYECANDY
