


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../Pipeline.h"
#include "EyeCandyMaterial.h"


namespace cuRE
{
	EyeCandyMaterial::EyeCandyMaterial(Pipeline& pipeline)
		: pipeline(pipeline)
	{
	}

	void EyeCandyMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.bindEyeCandyPipelineKernel();
		geometry->draw();
	}

	void EyeCandyMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.bindEyeCandyPipelineKernel();
		geometry->draw(start, num_indices);
	}


	VertexHeavyEyeCandyMaterial::VertexHeavyEyeCandyMaterial(Pipeline& pipeline, int iterations)
		: pipeline(pipeline), iterations(iterations)
	{
	}

	void VertexHeavyEyeCandyMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindVertexHeavyEyeCandyPipelineKernel();
		geometry->draw();
	}

	void VertexHeavyEyeCandyMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindVertexHeavyEyeCandyPipelineKernel();
		geometry->draw(start, num_indices);
	}


	FragmentHeavyEyeCandyMaterial::FragmentHeavyEyeCandyMaterial(Pipeline& pipeline, int iterations)
		: pipeline(pipeline), iterations(iterations)
	{
	}

	void FragmentHeavyEyeCandyMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindFragmentHeavyEyeCandyPipelineKernel();
		geometry->draw();
	}

	void FragmentHeavyEyeCandyMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindFragmentHeavyEyeCandyPipelineKernel();
		geometry->draw(start, num_indices);
	}
}
