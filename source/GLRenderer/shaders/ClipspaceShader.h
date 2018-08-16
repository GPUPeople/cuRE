


#ifndef INCLUDED_GLRENDERER_SHADER_CLIPSPACE
#define INCLUDED_GLRENDERER_SHADER_CLIPSPACE

#include <GL/texture.h>
#include <GL/shader.h>

#include <Resource.h>


namespace GLRenderer
{
	class ClipspaceShader
	{
		ClipspaceShader(const ClipspaceShader&) = delete;
		ClipspaceShader& operator =(const ClipspaceShader&) = delete;

		GL::Program prog;

	public:
		ClipspaceShader();

		void draw(const ::Geometry* geometry) const;
	};

	class VertexHeavyClipspaceShader
	{
		VertexHeavyClipspaceShader(const VertexHeavyClipspaceShader&) = delete;
		VertexHeavyClipspaceShader& operator =(const VertexHeavyClipspaceShader&) = delete;

		GL::Program heavy;
		GL::Program super_heavy;

	public:
		VertexHeavyClipspaceShader();

		void draw(const ::Geometry* geometry) const;
		void drawSuperHeavy(const ::Geometry* geometry) const;
	};

	class FragmentHeavyClipspaceShader
	{
		FragmentHeavyClipspaceShader(const FragmentHeavyClipspaceShader&) = delete;
		FragmentHeavyClipspaceShader& operator =(const FragmentHeavyClipspaceShader&) = delete;

		GL::Program heavy;
		GL::Program super_heavy;

	public:
		FragmentHeavyClipspaceShader();

		void draw(const ::Geometry* geometry) const;
		void drawSuperHeavy(const ::Geometry* geometry) const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_CLIPSPACE
