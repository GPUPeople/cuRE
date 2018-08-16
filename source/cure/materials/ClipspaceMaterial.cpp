


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../Pipeline.h"
#include "ClipspaceMaterial.h"


namespace cuRE
{
	ClipspaceMaterial::ClipspaceMaterial(Pipeline& pipeline)
		: pipeline(pipeline)
	{
	}

	void ClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.bindClipspacePipelineKernel();
		geometry->draw();
	}

	void ClipspaceMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.bindClipspacePipelineKernel();
		geometry->draw(start, num_indices);
	}


	VertexHeavyClipspaceMaterial::VertexHeavyClipspaceMaterial(Pipeline& pipeline, int iterations)
		: pipeline(pipeline), iterations(iterations)
	{
	}

	void VertexHeavyClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindVertexHeavyClipspacePipelineKernel();
		geometry->draw();
	}

	void VertexHeavyClipspaceMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindVertexHeavyClipspacePipelineKernel();
		geometry->draw(start, num_indices);
	}


	FragmentHeavyClipspaceMaterial::FragmentHeavyClipspaceMaterial(Pipeline& pipeline, int iterations)
		: pipeline(pipeline), iterations(iterations)
	{
	}

	void FragmentHeavyClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindFragmentHeavyClipspacePipelineKernel();
		geometry->draw();
	}

	void FragmentHeavyClipspaceMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setUniformi(5, iterations);
		pipeline.bindFragmentHeavyClipspacePipelineKernel();
		geometry->draw(start, num_indices);
	}
}
