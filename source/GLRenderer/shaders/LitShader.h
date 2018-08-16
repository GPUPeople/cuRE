


#ifndef INCLUDED_GLRENDERER_SHADER_LIT
#define INCLUDED_GLRENDERER_SHADER_LIT

#include <GL/shader.h>

#include <Resource.h>


namespace GLRenderer
{
	class LitShader
	{
		LitShader(const LitShader&) = delete;
		LitShader& operator =(const LitShader&) = delete;

		GL::VertexShader vs;
		GL::FragmentShader fs;
		GL::Program prog;

	public:
		LitShader();

		void draw(const ::Geometry* geometry, const math::float4& color) const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_LIT
