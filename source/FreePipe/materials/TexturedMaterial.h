


#ifndef INCLUDED_FREEPIPE_MATERIAL_TEXTURED
#define INCLUDED_FREEPIPE_MATERIAL_TEXTURED

#include <CUDA/module.h>

#include <Resource.h>


namespace FreePipe
{
	class Renderer;
	class Texture;

	class TexturedMaterial : public Material
	{
		TexturedMaterial(const TexturedMaterial&) = delete;
		TexturedMaterial& operator =(const TexturedMaterial&) = delete;

		Renderer& renderer;
		Texture& tex;

		CUdeviceptr mat_color;
		CUfunction kernel_fragment_processing;
		CUtexref texref;

		math::float4 color;

		CUmodule module;

	public:
		TexturedMaterial(Renderer& renderer, Texture& tex, const math::float4& color, CUmodule module);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num) const override;
	};
}

#endif  // INCLUDED_FREEPIPE_MATERIAL_TEXTURED
