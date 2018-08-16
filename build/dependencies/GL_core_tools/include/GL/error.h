


#ifndef INCLUDED_GL_ERROR
#define INCLUDED_GL_ERROR

#pragma once

#include <exception>

#include <GL/gl.h>


namespace GL
{
	class error : public std::exception
	{
	public:
		virtual GLenum code() const noexcept = 0;
		virtual const char* name() const noexcept = 0;
	};

	class context_lost : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class invalid_enum : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class invalid_value : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class invalid_operation : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class invalid_framebuffer_operation : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class out_of_memory : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class stack_overflow : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class stack_underflow : public error
	{
	public:
		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};

	class unknown_error : public error
	{
		GLenum error_code;

	public:
		unknown_error(GLenum error_code);

		GLenum code() const noexcept override;
		const char* name() const noexcept override;
		const char* what() const noexcept override;
	};


	GLenum throw_error(GLenum error);

	inline void throw_error()
	{
		GLenum error = glGetError();
		if (error != GL_NO_ERROR)
			throw unknown_error(throw_error(error));
	}
}

#endif  // INCLUDED_GL_ERROR
