


#include <GL/error.h>


namespace GL
{
	GLenum context_lost::code() const noexcept
	{
		return GL_CONTEXT_LOST;
	}

	const char* context_lost::name() const noexcept
	{
		return "GL_CONTEXT_LOST";
	}

	const char* context_lost::what() const noexcept
	{
		return "Context has been lost and reset by the driver";
	}


	GLenum invalid_enum::code() const noexcept
	{
		return GL_INVALID_ENUM;
	}

	const char* invalid_enum::name() const noexcept
	{
		return "GL_INVALID_ENUM";
	}

	const char* invalid_enum::what() const noexcept
	{
		return "enum argument out of range";
	}


	GLenum invalid_value::code() const noexcept
	{
		return GL_INVALID_VALUE;
	}

	const char* invalid_value::name() const noexcept
	{
		return "GL_INVALID_VALUE";
	}

	const char* invalid_value::what() const noexcept
	{
		return "Numeric argument out of range";
	}


	GLenum invalid_operation::code() const noexcept
	{
		return GL_INVALID_OPERATION;
	}

	const char* invalid_operation::name() const noexcept
	{
		return "GL_INVALID_OPERATION";
	}

	const char* invalid_operation::what() const noexcept
	{
		return "Operation illegal in current state";
	}


	GLenum invalid_framebuffer_operation::code() const noexcept
	{
		return GL_INVALID_FRAMEBUFFER_OPERATION;
	}

	const char* invalid_framebuffer_operation::name() const noexcept
	{
		return "GL_INVALID_FRAMEBUFFER_OPERATION";
	}

	const char* invalid_framebuffer_operation::what() const noexcept
	{
		return "Framebuffer object is not complete";
	}


	GLenum out_of_memory::code() const noexcept
	{
		return GL_OUT_OF_MEMORY;
	}

	const char* out_of_memory::name() const noexcept
	{
		return "GL_OUT_OF_MEMORY";
	}

	const char* out_of_memory::what() const noexcept
	{
		return "Not enough memory left to execute command";
	}


	GLenum stack_overflow::code() const noexcept
	{
		return GL_STACK_OVERFLOW;
	}

	const char* stack_overflow::name() const noexcept
	{
		return "GL_STACK_OVERFLOW";
	}

	const char* stack_overflow::what() const noexcept
	{
		return "Command would cause a stack overflow";
	}


	GLenum stack_underflow::code() const noexcept
	{
		return GL_STACK_UNDERFLOW;
	}

	const char* stack_underflow::name() const noexcept
	{
		return "GL_STACK_UNDERFLOW";
	}

	const char* stack_underflow::what() const noexcept
	{
		return "Command would cause a stack underflow";
	}


	unknown_error::unknown_error(GLenum error_code)
		: error_code(error_code)
	{
	}

	GLenum unknown_error::code() const noexcept
	{
		return error_code;
	}

	const char* unknown_error::name() const noexcept
	{
		return "ERROR";
	}

	const char* unknown_error::what() const noexcept
	{
		return "unknown error code";
	}


	GLenum throw_error(GLenum error)
	{
		switch (error)
		{
		case GL_CONTEXT_LOST:
			throw context_lost();
		case GL_INVALID_ENUM:
			throw invalid_enum();
		case GL_INVALID_VALUE:
			throw invalid_value();
		case GL_INVALID_OPERATION:
			throw invalid_operation();
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			throw invalid_framebuffer_operation();
		case GL_OUT_OF_MEMORY:
			throw out_of_memory();
		case GL_STACK_OVERFLOW:
			throw stack_overflow();
		case GL_STACK_UNDERFLOW:
			throw stack_underflow();
		}

		return error;
	}
}
