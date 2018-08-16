


#ifndef INCLUDED_GLRENDERER_SHADER_CLIPSPACEWIREFRAME
#define INCLUDED_GLRENDERER_SHADER_CLIPSPACEWIREFRAME

#include <GL/texture.h>
#include <GL/shader.h>

#include <Resource.h>


namespace GLRenderer
{
	class ClipspaceWireframeShader
	{
		ClipspaceWireframeShader(const ClipspaceWireframeShader&) = delete;
		ClipspaceWireframeShader& operator =(const ClipspaceWireframeShader&) = delete;

		GL::VertexShader vs;
		GL::GeometryShader gs;
		GL::FragmentShader fs;
		GL::Program prog;

	public:
		ClipspaceWireframeShader();

		void draw(const ::Geometry* geometry) const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_CLIPSPACEWIREFRAME
