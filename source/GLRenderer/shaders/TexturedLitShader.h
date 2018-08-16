


#ifndef INCLUDED_GLRENDERER_SHADER_TEXTURED_LIT
#define INCLUDED_GLRENDERER_SHADER_TEXTURED_LIT

#include <GL/texture.h>
#include <GL/shader.h>

#include <Resource.h>


namespace GLRenderer
{
	class TexturedLitShader
	{
		TexturedLitShader(const TexturedLitShader&) = delete;
		TexturedLitShader& operator =(const TexturedLitShader&) = delete;

		GL::VertexShader vs;
		GL::FragmentShader fs;
		GL::Program prog;

		GL::Sampler sampler;

	public:
		TexturedLitShader();

		void draw(const ::Geometry* geometry, GLuint texture, const math::float4& color) const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_TEXTURED_LIT
