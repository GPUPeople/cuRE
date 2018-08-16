


#ifndef INCLUDED_CURE_MATERIAL_LIT
#define INCLUDED_CURE_MATERIAL_LIT

#include <CUDA/module.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class LitMaterial : public Material
	{
		LitMaterial(const LitMaterial&) = delete;
		LitMaterial& operator =(const LitMaterial&) = delete;

		Pipeline& pipeline;

	public:
		LitMaterial(Pipeline& pipeline, const math::float4& color);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_CURE_MATERIAL_LIT
