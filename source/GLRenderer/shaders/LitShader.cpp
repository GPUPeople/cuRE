


#include "LitShader.h"


extern const char lit_vs[];
extern const char lit_fs[];

namespace GLRenderer
{
	LitShader::LitShader()
		: vs(GL::compileVertexShader(lit_vs)),
		  fs(GL::compileFragmentShader(lit_fs))
	{
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void LitShader::draw(const ::Geometry* geometry, const math::float4& color) const
	{
		glUseProgram(prog);
		glUniform4fv(0, 1, &color.x);
		geometry->draw();
	}
}
