


#include "../config.h"

#include "EyeCandyShader.h"


extern const char eyecandy_vs[];
extern const char eyecandy_fs[];
extern const char eyecandy_interlocked_fs[];

extern const char eyecandy_vertex_heavy_vs[];
extern const char eyecandy_vertex_heavy_fs[];
extern const char eyecandy_vertex_heavy_interlocked_fs[];

extern const char eyecandy_fragment_heavy_vs[];
extern const char eyecandy_fragment_heavy_fs[];
extern const char eyecandy_fragment_heavy_interlocked_fs[];

namespace GLRenderer
{
	EyeCandyShader::EyeCandyShader()
	{
		auto vs = GL::compileVertexShader(eyecandy_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? eyecandy_interlocked_fs : eyecandy_fs);
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void EyeCandyShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(prog);
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		geometry->draw();
		//glDisable(GL_BLEND);
	}


	VertexHeavyEyeCandyShader::VertexHeavyEyeCandyShader()
	{
		auto vs = GL::compileVertexShader(eyecandy_vertex_heavy_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? eyecandy_vertex_heavy_interlocked_fs : eyecandy_vertex_heavy_fs);
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void VertexHeavyEyeCandyShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(prog);
		geometry->draw();
	}


	FragmentHeavyEyeCandyShader::FragmentHeavyEyeCandyShader()
	{
		auto vs = GL::compileVertexShader(eyecandy_fragment_heavy_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? eyecandy_fragment_heavy_interlocked_fs : eyecandy_fragment_heavy_fs);
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void FragmentHeavyEyeCandyShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(prog);
		geometry->draw();
	}
}
