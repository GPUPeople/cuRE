


#ifndef INCLUDED_FREEPIPE_MATERIAL_COLORED
#define INCLUDED_FREEPIPE_MATERIAL_COLORED

#include <CUDA/module.h>

#include <Resource.h>


namespace FreePipe
{
	class Renderer;

	class ColoredMaterial : public Material
	{
		ColoredMaterial(const ColoredMaterial&) = delete;
		ColoredMaterial& operator =(const ColoredMaterial&) = delete;

		CUdeviceptr color_data;
		CUfunction kernel_fragment_processing;

		math::float4 color;

		CUmodule module;

		Renderer& renderer;

	public:
		ColoredMaterial(Renderer& renderer, const math::float4& color, CUmodule module);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num) const override;
	};
}

#endif  // INCLUDED_FREEPIPE_MATERIAL_LIT
