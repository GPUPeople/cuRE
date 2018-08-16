


#include <GL/error.h>
#include <GL/vertex_array.h>


namespace GL
{
	GLuint VertexArrayObjectNamespace::gen()
	{
		GLuint id;
		glGenVertexArrays(1, &id);
		throw_error();
		return id;
	}

	void VertexArrayObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteVertexArrays(1, &name);
	}
}
