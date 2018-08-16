


#include <GL/error.h>
#include <GL/event.h>


namespace GL
{
	GLuint QueryObjectNamespace::gen()
	{
		GLuint name;
		glGenQueries(1, &name);
		throw_error();
		return name;
	}

	void QueryObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteQueries(1, &name);
	}
}
