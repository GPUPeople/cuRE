


#include "../config.h"

#include "ClipspaceShader.h"


extern const char clipspace_vs[];
extern const char clipspace_fs[];
extern const char clipspace_interlocked_fs[];

extern const char clipspace_vertex_heavy_vs[];
extern const char clipspace_vertex_heavy_fs[];
extern const char clipspace_vertex_heavy_interlocked_fs[];

extern const char clipspace_fragment_heavy_fs[];
extern const char clipspace_fragment_heavy_interlocked_fs[];


namespace GLRenderer
{
	ClipspaceShader::ClipspaceShader()
	{
		auto vs = GL::compileVertexShader(clipspace_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? clipspace_interlocked_fs : clipspace_fs);
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void ClipspaceShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(prog);
		geometry->draw();
	}


	VertexHeavyClipspaceShader::VertexHeavyClipspaceShader()
	{
		auto vs = GL::compileVertexShader(clipspace_vertex_heavy_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? clipspace_vertex_heavy_interlocked_fs : clipspace_vertex_heavy_fs);
		glAttachShader(heavy, vs);
		glAttachShader(heavy, fs);
		GL::linkProgram(heavy);
	}

	void VertexHeavyClipspaceShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(heavy);
		geometry->draw();
	}

	void VertexHeavyClipspaceShader::drawSuperHeavy(const ::Geometry* geometry) const
	{
		glUseProgram(heavy);
		geometry->draw();
	}


	FragmentHeavyClipspaceShader::FragmentHeavyClipspaceShader()
	{
		auto vs = GL::compileVertexShader(clipspace_vs);
		auto fs = GL::compileFragmentShader(FRAGMENT_SHADER_INTERLOCK ? clipspace_fragment_heavy_interlocked_fs : clipspace_fragment_heavy_fs);
		glAttachShader(heavy, vs);
		glAttachShader(heavy, fs);
		GL::linkProgram(heavy);
	}

	void FragmentHeavyClipspaceShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(heavy);
		geometry->draw();
	}

	void FragmentHeavyClipspaceShader::drawSuperHeavy(const ::Geometry* geometry) const
	{
		glUseProgram(heavy);
		geometry->draw();
	}
}
