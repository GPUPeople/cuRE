


#include <GL/error.h>
#include <GL/buffer.h>


namespace GL
{
	GLuint BufferObjectNamespace::gen()
	{
		GLuint id;
		glGenBuffers(1, &id);
		throw_error();
		return id;
	}

	void BufferObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteBuffers(1, &name);
	}
}
