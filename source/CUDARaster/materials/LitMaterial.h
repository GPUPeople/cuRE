


#ifndef INCLUDED_CUDARASTER_MATERIAL_LIT
#define INCLUDED_CUDARASTER_MATERIAL_LIT

#include <CUDA/module.h>
#include "../Material.h"
#include "../Shaders.hpp"

namespace CUDARaster
{
	class LitMaterial : public CUDARaster::Material
	{
		LitMaterial(const LitMaterial&) = delete;
		LitMaterial& operator =(const LitMaterial&) = delete;

		CUdeviceptr phongData;
		CUmodule module;
		CUfunction kernel_fragment_processing;

		CUdeviceptr matData;
		FW::Material matbase;

	public:
		LitMaterial(const math::float4& color, CUmodule module);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;

		void setModel(const math::float3x4& mat);
		void setCamera(const math::float4x4& PV, const math::float3& pos);
		void remove()
		{
		}

	};
}

#endif  // INCLUDED_CUDARASTER_MATERIAL_LIT
