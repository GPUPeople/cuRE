


#include "GLRenderer/shaders/HeavyShaders.h"
#include "HeavyMaterials.h"


GL::Buffer createNoiseParametersBuffer(GLuint iterations)
{
	GL::Buffer buffer;
	glBindBuffer(GL_UNIFORM_BUFFER, buffer);
	glBufferStorage(GL_UNIFORM_BUFFER, sizeof(iterations), &iterations, 0U);
	return buffer;
}

namespace GLRenderer
{
	VertexHeavyMaterial::VertexHeavyMaterial(const HeavyVertexShader& shader, int iterations)
		: shader(shader), noise_params(createNoiseParametersBuffer(iterations))
	{
	}

	void VertexHeavyMaterial::draw(const ::Geometry* geometry) const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 5U, noise_params);
		shader.draw(geometry);
	}

	void VertexHeavyMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}

	FragmentHeavyMaterial::FragmentHeavyMaterial(const HeavyFragmentShader& shader, int iterations)
		: shader(shader), noise_params(createNoiseParametersBuffer(iterations))
	{
	}

	void FragmentHeavyMaterial::draw(const ::Geometry* geometry) const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 5U, noise_params);
		shader.draw(geometry);
	}

	void FragmentHeavyMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
