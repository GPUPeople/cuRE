


#ifndef INCLUDED_GLRENDERER_SHADER_TEXTURED
#define INCLUDED_GLRENDERER_SHADER_TEXTURED

#include <GL/texture.h>
#include <GL/shader.h>

#include <Resource.h>


namespace GLRenderer
{
	class TexturedShader
	{
		TexturedShader(const TexturedShader&) = delete;
		TexturedShader& operator =(const TexturedShader&) = delete;

		GL::VertexShader vs;
		GL::FragmentShader fs;
		GL::Program prog;

		GL::Sampler sampler;

	public:
		TexturedShader();

		void draw(const ::Geometry* geometry, GLuint texture, const math::float4& color) const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_TEXTURED
