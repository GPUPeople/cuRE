


#include "ClipspaceWireframeShader.h"


extern const char clipspace_wireframe_vs[];
extern const char clipspace_wireframe_gs[];
extern const char clipspace_wireframe_fs[];

namespace GLRenderer
{
	ClipspaceWireframeShader::ClipspaceWireframeShader()
		: vs(GL::compileVertexShader(clipspace_wireframe_vs)),
		  gs(GL::compileGeometryShader(clipspace_wireframe_gs)),
		  fs(GL::compileFragmentShader(clipspace_wireframe_fs))
	{
		glAttachShader(prog, vs);
		glAttachShader(prog, gs);
		glAttachShader(prog, fs);
		GL::linkProgram(prog);
	}

	void ClipspaceWireframeShader::draw(const ::Geometry* geometry) const
	{
		glUseProgram(prog);
		geometry->draw();
		glDisable(GL_BLEND);
	}
}
