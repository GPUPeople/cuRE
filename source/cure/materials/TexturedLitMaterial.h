


#ifndef INCLUDED_CURE_MATERIAL_TEXTURED_LIT
#define INCLUDED_CURE_MATERIAL_TEXTURED_LIT

#include <cuda.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class TexturedLitMaterial : public Material
	{
		TexturedLitMaterial(const TexturedLitMaterial&) = delete;
		TexturedLitMaterial& operator =(const TexturedLitMaterial&) = delete;

		Pipeline& pipeline;
		CUmipmappedArray tex;

	public:
		TexturedLitMaterial(Pipeline& pipeline, CUmipmappedArray tex, const math::float4& color);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_CURE_MATERIAL_TEXTURED_LIT
