


#ifndef INCLUDED_GLRENDERER_SHADERS_HEAVY
#define INCLUDED_GLRENDERER_SHADERS_HEAVY

#include <GL/shader.h>

#include <Resource.h>


namespace GLRenderer
{
	class HeavyVertexShader
	{
		HeavyVertexShader(const HeavyVertexShader&) = delete;
		HeavyVertexShader& operator =(const HeavyVertexShader&) = delete;

		GL::Program prog;

	public:
		HeavyVertexShader();

		void draw(const ::Geometry* geometry) const;
	};

	class HeavyFragmentShader
	{
		HeavyFragmentShader(const HeavyFragmentShader&) = delete;
		HeavyFragmentShader& operator =(const HeavyFragmentShader&) = delete;

		GL::Program prog;

	public:
		HeavyFragmentShader();

		void draw(const ::Geometry* geometry) const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADERS_HEAVY
