


#ifndef INCLUDED_GLRENDERER_MATERIAL_TEXTURED
#define INCLUDED_GLRENDERER_MATERIAL_TEXTURED

#include <GL/gl.h>

#include <Resource.h>


namespace GLRenderer
{
	class TexturedShader;

	class TexturedMaterial : public ::Material
	{
		TexturedMaterial(const TexturedMaterial&) = delete;
		TexturedMaterial& operator =(const TexturedMaterial&) = delete;

		const TexturedShader& shader;

		GLuint texture;
		math::float4 color;

	public:
		TexturedMaterial(const TexturedShader& shader, GLuint texture, const math::float4& color);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_GLRENDERER_MATERIAL_TEXTURED
