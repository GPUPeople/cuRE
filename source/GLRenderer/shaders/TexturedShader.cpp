


#include <GL/error.h>
#include "TexturedShader.h"


extern const char textured_vs[];
extern const char textured_fs[];

namespace GLRenderer
{
	TexturedShader::TexturedShader()
		: vs(GL::compileVertexShader(textured_vs)),
		  fs(GL::compileFragmentShader(textured_fs))
	{
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
		GL::throw_error();
	}

	void TexturedShader::draw(const ::Geometry* geometry, GLuint texture, const math::float4& color) const
	{
		glUseProgram(prog);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glBindSampler(0, sampler);
		glUniform4fv(0, 1, &color.x);
		geometry->draw();
	}
}
