


#include <GL/error.h>
#include <GL/framebuffer.h>


namespace GL
{
	GLuint FramebufferObjectNamespace::gen()
	{
		GLuint id;
		glGenFramebuffers(1, &id);
		throw_error();
		return id;
	}

	void FramebufferObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteFramebuffers(1, &name);
	}


	GLuint RenderbufferObjectNamespace::gen()
	{
		GLuint id;
		glGenRenderbuffers(1, &id);
		throw_error();
		return id;
	}

	void RenderbufferObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteRenderbuffers(1, &name);
	}
	

	Renderbuffer createRenderbuffer(GLsizei width, GLsizei height, GLenum format)
	{
		Renderbuffer renderbuffer;
		glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
		throw_error();
		return renderbuffer;
	}

	Renderbuffer createRenderbuffer(GLsizei width, GLsizei height, GLenum format, GLsizei samples)
	{
		Renderbuffer renderbuffer;
		glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, format, width, height);
		throw_error();
		return renderbuffer;
	}
}
