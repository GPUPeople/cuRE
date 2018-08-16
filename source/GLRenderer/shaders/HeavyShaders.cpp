


#include "../config.h"

#include "HeavyShaders.h"


extern const char heavy_vertex_vs[];
extern const char heavy_vertex_fs[];
extern const char heavy_vertex_interlocked_fs[];
extern const char heavy_fragment_vs[];
extern const char heavy_fragment_fs[];
extern const char heavy_fragment_interlocked_fs[];

namespace GLRenderer
{
	HeavyVertexShader::HeavyVertexShader()
	{
		auto vs = GL::compileVertexShader(heavy_vertex_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? heavy_vertex_interlocked_fs : heavy_vertex_fs);
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void HeavyVertexShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(prog);
		geometry->draw();
	}

	HeavyFragmentShader::HeavyFragmentShader()
	{
		auto vs = GL::compileVertexShader(heavy_fragment_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? heavy_fragment_interlocked_fs : heavy_fragment_fs);
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void HeavyFragmentShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(prog);
		geometry->draw();
	}
}
