


#ifndef INCLUDED_GLRENDERER_MATERIAL_COLORED
#define INCLUDED_GLRENDERER_MATERIAL_COLORED

#include <Resource.h>


namespace GLRenderer
{
	class ColoredShader;

	class ColoredMaterial : public ::Material
	{
	protected:
		ColoredMaterial(const ColoredMaterial&) = delete;
		ColoredMaterial& operator =(const ColoredMaterial&) = delete;

		const ColoredShader& shader;

		math::float4 color;

	public:
		ColoredMaterial(const ColoredShader& shader, const math::float4& color);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_GLRENDERER_MATERIAL_COLORED
