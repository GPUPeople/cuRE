


#include <GL/error.h>

#include "TexturedLitShader.h"


extern const char textured_lit_vs[];
extern const char textured_lit_fs[];

namespace GLRenderer
{
	TexturedLitShader::TexturedLitShader()
		: vs(GL::compileVertexShader(textured_lit_vs)),
		  fs(GL::compileFragmentShader(textured_lit_fs))
	{
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);

		glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		GL::throw_error();
	}

	void TexturedLitShader::draw(const ::Geometry* geometry, GLuint texture, const math::float4& color) const
	{
		glUseProgram(prog);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glBindSampler(0, sampler);
		glUniform4fv(0, 1, &color.x);
		geometry->draw();
	}
}
