


#ifndef INCLUDED_GLRENDERER_MATERIAL_LIT
#define INCLUDED_GLRENDERER_MATERIAL_LIT

#include <Resource.h>


namespace GLRenderer
{
	class LitShader;

	class LitMaterial : public ::Material
	{
		LitMaterial(const LitMaterial&) = delete;
		LitMaterial& operator =(const LitMaterial&) = delete;

		const LitShader& shader;

		math::float4 color;

	public:
		LitMaterial(const LitShader& shader, const math::float4& color);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_GLRENDERER_MATERIAL_LIT
