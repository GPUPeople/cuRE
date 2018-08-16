


#include "GLRenderer/shaders/TexturedLitShader.h"
#include "TexturedLitMaterial.h"


namespace GLRenderer
{
	TexturedLitMaterial::TexturedLitMaterial(const TexturedLitShader& shader, GLuint texture, const math::float4& color)
		: shader(shader),
		  texture(texture),
		  color(color)
	{
	}

	void TexturedLitMaterial::draw(const ::Geometry* geometry) const
	{
		shader.draw(geometry, texture, color);
	}

	void TexturedLitMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
