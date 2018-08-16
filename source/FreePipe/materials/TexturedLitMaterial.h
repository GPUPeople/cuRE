


#ifndef INCLUDED_FREEPIPE_MATERIAL_TEXTURED_LIT
#define INCLUDED_FREEPIPE_MATERIAL_TEXTURED_LIT

#include <CUDA/module.h>

#include <Resource.h>


namespace FreePipe
{
	class Renderer;
	class Texture;

	class TexturedLitMaterial : public Material
	{
		TexturedLitMaterial(const TexturedLitMaterial&) = delete;
		TexturedLitMaterial& operator =(const TexturedLitMaterial&) = delete;

		Renderer& renderer;
		Texture& tex;

		CUdeviceptr phong_data;
		CUdeviceptr light_data;
		CUfunction kernel_fragment_processing;
		CUtexref texref;

		math::float4 color;

		CUmodule module;

	public:
		TexturedLitMaterial(Renderer& renderer, Texture& tex, const math::float4& color, CUmodule module);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num) const override;
	};
}

#endif  // INCLUDED_FREEPIPE_MATERIAL_TEXTURED_LIT
