


#include <limits>

#include "../Pipeline.h"
#include "HeavyMaterial.h"


namespace cuRE
{
	VertexHeavyMaterial::VertexHeavyMaterial(Pipeline& pipeline, int iterations)
		: pipeline(pipeline), iterations(iterations)
	{
	}

	void VertexHeavyMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindVertexHeavyPipelineKernel();
		geometry->draw();
	}

	void VertexHeavyMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindVertexHeavyPipelineKernel();
		geometry->draw(start, num_indices);
	}


	FragmentHeavyMaterial::FragmentHeavyMaterial(Pipeline& pipeline, int iterations)
		: pipeline(pipeline), iterations(iterations)
	{
	}

	void FragmentHeavyMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindFragmentHeavyPipelineKernel();
		geometry->draw();
	}

	void FragmentHeavyMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindFragmentHeavyPipelineKernel();
		geometry->draw(start, num_indices);
	}
}
