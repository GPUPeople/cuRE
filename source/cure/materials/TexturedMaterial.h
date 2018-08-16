


#ifndef INCLUDED_CURE_MATERIAL_TEXTURED
#define INCLUDED_CURE_MATERIAL_TEXTURED

#include <CUDA/module.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class TexturedMaterial : public Material
	{
		TexturedMaterial(const TexturedMaterial&) = delete;
		TexturedMaterial& operator =(const TexturedMaterial&) = delete;

		Pipeline& pipeline;

		CUmipmappedArray tex;

	public:
		TexturedMaterial(Pipeline& pipeline, CUmipmappedArray tex, const math::float4& color);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_CURE_MATERIAL_TEXTURED
