


#include <limits>

#include "../Pipeline.h"

#include "TexturedMaterial.h"


namespace cuRE
{
	TexturedMaterial::TexturedMaterial(Pipeline& pipeline, CUmipmappedArray tex, const math::float4& color)
		: pipeline(pipeline),
		  tex(tex)
	{
	}

	void TexturedMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setTextureSRGB(tex, std::numeric_limits<float>::max());
		pipeline.bindTexturedPipelineKernel();
		geometry->draw();
	}

	void TexturedMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setTextureSRGB(tex, std::numeric_limits<float>::max());
		pipeline.bindTexturedPipelineKernel();
		geometry->draw(start, num_indices);
	}
}
