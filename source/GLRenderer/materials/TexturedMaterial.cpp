


#include "GLRenderer/shaders/TexturedShader.h"
#include "TexturedMaterial.h"


namespace GLRenderer
{
	TexturedMaterial::TexturedMaterial(const TexturedShader& shader, GLuint texture, const math::float4& color)
		: shader(shader),
		  texture(texture),
		  color(color)
	{
	}

	void TexturedMaterial::draw(const ::Geometry* geometry) const
	{
		shader.draw(geometry, texture, color);
	}

	void TexturedMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
