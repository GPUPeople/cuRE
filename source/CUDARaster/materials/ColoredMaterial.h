


#ifndef INCLUDED_CUDARASTER_MATERIAL_COL
#define INCLUDED_CUDARASTER_MATERIAL_COL

#include <CUDA/module.h>
#include "../Material.h"
#include "../Shaders.hpp"

namespace CUDARaster
{
	class ColoredMaterial : public CUDARaster::Material
	{
		ColoredMaterial(const ColoredMaterial&) = delete;
		ColoredMaterial& operator =(const ColoredMaterial&) = delete;

		CUdeviceptr phongData;
		CUmodule module;
		CUfunction kernel_fragment_processing;

		CUdeviceptr matData;
		FW::Material matbase;

	public:
		ColoredMaterial(const math::float4& color, CUmodule module);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;

		void setModel(const math::float3x4& mat);
		void setCamera(const math::float4x4& PV, const math::float3& pos);
		void remove()
		{
		}

	};
}

#endif  // INCLUDED_CUDARASTER_MATERIAL_COL
