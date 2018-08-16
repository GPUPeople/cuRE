


#ifndef INCLUDED_GLRENDERER_MATERIAL_TEXTURED_LIT
#define INCLUDED_GLRENDERER_MATERIAL_TEXTURED_LIT

#include <GL/gl.h>

#include <Resource.h>


namespace GLRenderer
{
	class TexturedLitShader;

	class TexturedLitMaterial : public ::Material
	{
		TexturedLitMaterial(const TexturedLitMaterial&) = delete;
		TexturedLitMaterial& operator =(const TexturedLitMaterial&) = delete;

		const TexturedLitShader& shader;

		GLuint texture;
		math::float4 color;

	public:
		TexturedLitMaterial(const TexturedLitShader& shader, GLuint texture, const math::float4& color);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_GLRENDERER_MATERIAL_TEXTURED_LIT
