


#ifndef INCLUDED_FREEPIPE_MATERIAL_LIT
#define INCLUDED_FREEPIPE_MATERIAL_LIT

#include <CUDA/module.h>

#include <Resource.h>


namespace FreePipe
{
	class Renderer;

	class LitMaterial : public Material
	{
		LitMaterial(const LitMaterial&) = delete;
		LitMaterial& operator =(const LitMaterial&) = delete;

		CUdeviceptr phong_data;
		CUdeviceptr light_data;
		CUfunction kernel_fragment_processing;

		math::float4 color;

		CUmodule module;

		Renderer& renderer;

	public:
		LitMaterial(Renderer& renderer, const math::float4& color, CUmodule module);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num) const override;
	};
}

#endif  // INCLUDED_FREEPIPE_MATERIAL_LIT
