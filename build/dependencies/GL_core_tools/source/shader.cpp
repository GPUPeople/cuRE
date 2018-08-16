


#include <string>

#include <GL/error.h>
#include <GL/shader.h>


namespace
{
	void compileShaderSource(GLuint shader, const char* source)
	{
		glShaderSource(shader, 1, &source, 0);
		glCompileShader(shader);
		GL::throw_error();
		if (GL::getShaderCompileStatus(shader) == false)
			throw GL::compile_error(GL::getShaderInfoLog(shader));
	}
}

namespace GL
{
	compile_error::compile_error(std::string log)
		: std::runtime_error(log)
	{
	}

	link_error::link_error(std::string log)
		: std::runtime_error(log)
	{
	}


	template <GLenum SHADER_TYPE>
	GLuint ShaderObjectNamespace<SHADER_TYPE>::gen()
	{
		GLuint name = glCreateShader(SHADER_TYPE);
		throw_error();
		return name;
	}

	template <GLenum ShaderType>
	void ShaderObjectNamespace<ShaderType>::del(GLuint name) noexcept
	{
		glDeleteShader(name);
	}

	template struct ShaderObjectNamespace<GL_VERTEX_SHADER>;
	template struct ShaderObjectNamespace<GL_TESS_CONTROL_SHADER>;
	template struct ShaderObjectNamespace<GL_TESS_EVALUATION_SHADER>;
	template struct ShaderObjectNamespace<GL_GEOMETRY_SHADER>;
	template struct ShaderObjectNamespace<GL_FRAGMENT_SHADER>;
	template struct ShaderObjectNamespace<GL_COMPUTE_SHADER>;


	bool getShaderCompileStatus(GLuint shader)
	{
		GLint b;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &b);
		throw_error();
		return b == GL_TRUE;
	}

	std::string getShaderInfoLog(GLuint shader)
	{
		GLint length;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
		throw_error();
		if (length == 0)
			return "";
		std::string log(length + 1, '\0');
		glGetShaderInfoLog(shader, length, 0, &log[0]);
		throw_error();
		return log;
	}

	VertexShader compileVertexShader(const char* source)
	{
		VertexShader shader;
		compileShaderSource(shader, source);
		return shader;
	}

	TessellationControlShader compileTessellationControlShader(const char* source)
	{
		TessellationControlShader shader;
		compileShaderSource(shader, source);
		return shader;
	}

	TessellationEvaluationShader compileTessellationEvaluationShader(const char* source)
	{
		TessellationEvaluationShader shader;
		compileShaderSource(shader, source);
		return shader;
	}

	GeometryShader compileGeometryShader(const char* source)
	{
		GeometryShader shader;
		compileShaderSource(shader, source);
		return shader;
	}

	FragmentShader compileFragmentShader(const char* source)
	{
		FragmentShader shader;
		compileShaderSource(shader, source);
		return shader;
	}

	ComputeShader compileComputeShader(const char* source)
	{
		ComputeShader shader;
		compileShaderSource(shader, source);
		return shader;
	}


	GLuint ProgramObjectNamespace::gen()
	{
		GLuint name;
		name = glCreateProgram();
		throw_error();
		return name;
	}

	void ProgramObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteProgram(name);
	}


	bool getProgramLinkStatus(GLuint program)
	{
		GLint b;
		glGetProgramiv(program, GL_LINK_STATUS, &b);
		throw_error();
		return b == GL_TRUE;
	}

	bool getProgramValidationStatus(GLuint program)
	{
		GLint b;
		glGetProgramiv(program, GL_VALIDATE_STATUS, &b);
		throw_error();
		return b == GL_TRUE;
	}

	std::string getProgramInfoLog(GLuint program)
	{
		GLint length;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
		throw_error();
		if (length == 0)
			return "";
		std::string log(length + 1, '\0');
		glGetProgramInfoLog(program, length, 0, &log[0]);
		throw_error();
		return log;
	}

	void linkProgram(GLuint program)
	{
		glLinkProgram(program);
		throw_error();
		if (getProgramLinkStatus(program) == false)
			throw link_error(getProgramInfoLog(program));
	}


	GLuint ProgramPipelineObjectNamespace::gen()
	{
		GLuint name;
		glGenProgramPipelines(1, &name);
		throw_error();
		return name;
	}

	void ProgramPipelineObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteProgramPipelines(1, &name);
	}
}
