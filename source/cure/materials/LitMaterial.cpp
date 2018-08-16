


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../Pipeline.h"
#include "LitMaterial.h"


namespace cuRE
{
	LitMaterial::LitMaterial(Pipeline& pipeline, const math::float4& color)
		: pipeline(pipeline)
	{
	}

	void LitMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.bindLitPipelineKernel();
		geometry->draw();
	}

	void LitMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.bindLitPipelineKernel();
		geometry->draw(start, num_indices);
	}
}
