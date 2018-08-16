


#ifndef INCLUDED_GLRENDERER_SHADER_COLORED
#define INCLUDED_GLRENDERER_SHADER_COLORED

#include <GL/shader.h>

#include <Resource.h>


namespace GLRenderer
{
	class ColoredShader
	{
		ColoredShader(const ColoredShader&) = delete;
		ColoredShader& operator =(const ColoredShader&) = delete;

		GL::VertexShader vs;
		GL::FragmentShader fs;
		GL::Program prog;

	public:
		ColoredShader();

		void draw(const ::Geometry* geometry, const math::float4& color) const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_COLORED
