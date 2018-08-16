


#include <limits>

#include "../Pipeline.h"

#include "TexturedLitMaterial.h"


namespace cuRE
{
	TexturedLitMaterial::TexturedLitMaterial(Pipeline& pipeline, CUmipmappedArray tex, const math::float4& color)
		: pipeline(pipeline),
		  tex(tex)
	{
	}

	void TexturedLitMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setTextureSRGB(tex, std::numeric_limits<float>::max());
		pipeline.bindTexturedPipelineKernel();
		geometry->draw();
	}

	void TexturedLitMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setTextureSRGB(tex, std::numeric_limits<float>::max());
		pipeline.bindTexturedPipelineKernel();
		geometry->draw(start, num_indices);
	}
}
