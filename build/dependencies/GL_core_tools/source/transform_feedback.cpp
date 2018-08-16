


#include <GL/error.h>
#include <GL/transform_feedback.h>


namespace GL
{
	GLuint TransformFeedbackObjectNamespace::gen()
	{
		GLuint id;
		glGenTransformFeedbacks(1, &id);
		throw_error();
		return id;
	}

	void TransformFeedbackObjectNamespace::del(GLuint name) noexcept
	{
		glDeleteTransformFeedbacks(1, &name);
	}
}
