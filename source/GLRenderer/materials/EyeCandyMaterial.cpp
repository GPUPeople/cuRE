


#include "GLRenderer/shaders/EyeCandyShader.h"
#include "EyeCandyMaterial.h"


GL::Buffer createNoiseParametersBuffer(GLuint iterations);

namespace GLRenderer
{
	EyeCandyMaterial::EyeCandyMaterial(const EyeCandyShader& shader)
		: shader(shader)
	{
	}

	void EyeCandyMaterial::draw(const ::Geometry* geometry) const
	{
		shader.draw(geometry);
	}

	void EyeCandyMaterial::draw(const::Geometry* geometry, int start, int num_indices) const
	{
	}


	VertexHeavyEyeCandyMaterial::VertexHeavyEyeCandyMaterial(const VertexHeavyEyeCandyShader& shader, int iterations)
		: shader(shader), noise_params(createNoiseParametersBuffer(iterations))
	{
	}

	void VertexHeavyEyeCandyMaterial::draw(const ::Geometry* geometry) const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 5U, noise_params);
		shader.draw(geometry);
	}

	void VertexHeavyEyeCandyMaterial::draw(const::Geometry* geometry, int start, int num_indices) const
	{
	}


	FragmentHeavyEyeCandyMaterial::FragmentHeavyEyeCandyMaterial(const FragmentHeavyEyeCandyShader& shader, int iterations)
		: shader(shader), noise_params(createNoiseParametersBuffer(iterations))
	{
	}

	void FragmentHeavyEyeCandyMaterial::draw(const ::Geometry* geometry) const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 5U, noise_params);
		shader.draw(geometry);
	}

	void FragmentHeavyEyeCandyMaterial::draw(const::Geometry* geometry, int start, int num_indices) const
	{
	}
}
