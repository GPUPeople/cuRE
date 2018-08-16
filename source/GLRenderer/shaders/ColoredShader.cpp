


#include "ColoredShader.h"


extern const char colored_vs[];
extern const char colored_fs[];

namespace GLRenderer
{
	ColoredShader::ColoredShader()
		: vs(GL::compileVertexShader(colored_vs)),
		  fs(GL::compileFragmentShader(colored_fs))
	{
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void ColoredShader::draw(const ::Geometry* geometry, const math::float4& color) const
	{
		glUseProgram(prog);
		glUniform4fv(0, 1, &color.x);
		geometry->draw();
	}
}
