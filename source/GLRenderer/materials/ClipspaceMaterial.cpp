


#include "GLRenderer/shaders/ClipspaceShader.h"
#include "ClipspaceMaterial.h"


GL::Buffer createNoiseParametersBuffer(GLuint iterations);

namespace GLRenderer
{
	ClipspaceMaterial::ClipspaceMaterial(const ClipspaceShader& shader)
		: shader(shader)
	{
	}

	void ClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		shader.draw(geometry);
	}

	void ClipspaceMaterial::draw(const::Geometry* geometry, int start, int num_indices) const
	{
	}


	VertexHeavyClipspaceMaterial::VertexHeavyClipspaceMaterial(const VertexHeavyClipspaceShader& shader, int iterations)
		: shader(shader), noise_params(createNoiseParametersBuffer(iterations))
	{
	}

	void VertexHeavyClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 5U, noise_params);
		shader.draw(geometry);
	}

	void VertexHeavyClipspaceMaterial::draw(const::Geometry* geometry, int start, int num_indices) const
	{
	}


	FragmentHeavyClipspaceMaterial::FragmentHeavyClipspaceMaterial(const FragmentHeavyClipspaceShader& shader, int iterations)
		: shader(shader), noise_params(createNoiseParametersBuffer(iterations))
	{
	}

	void FragmentHeavyClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 5U, noise_params);
		shader.draw(geometry);
	}

	void FragmentHeavyClipspaceMaterial::draw(const::Geometry* geometry, int start, int num_indices) const
	{
	}
}
