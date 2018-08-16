


#ifndef INCLUDED_GLRENDERER_SHADER_EYECANDY
#define INCLUDED_GLRENDERER_SHADER_EYECANDY

#include <GL/shader.h>
#include <GL/texture.h>

#include <Resource.h>


namespace GLRenderer
{
	class EyeCandyShader
	{
		EyeCandyShader(const EyeCandyShader&) = delete;
		EyeCandyShader& operator=(const EyeCandyShader&) = delete;

		GL::Program prog;

	public:
		EyeCandyShader();

		void draw(const ::Geometry* geometry) const;
	};

	class VertexHeavyEyeCandyShader
	{
		VertexHeavyEyeCandyShader(const VertexHeavyEyeCandyShader&) = delete;
		VertexHeavyEyeCandyShader& operator=(const VertexHeavyEyeCandyShader&) = delete;

		GL::Program prog;

	public:
		VertexHeavyEyeCandyShader();

		void draw(const ::Geometry* geometry) const;
	};

	class FragmentHeavyEyeCandyShader
	{
		FragmentHeavyEyeCandyShader(const FragmentHeavyEyeCandyShader&) = delete;
		FragmentHeavyEyeCandyShader& operator=(const FragmentHeavyEyeCandyShader&) = delete;

		GL::Program prog;

	public:
		FragmentHeavyEyeCandyShader();

		void draw(const ::Geometry* geometry) const;
	};
}

#endif // INCLUDED_GLRENDERER_SHADER_CLIPSPACE
