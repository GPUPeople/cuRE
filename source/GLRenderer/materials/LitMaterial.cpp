


#include "GLRenderer/shaders/LitShader.h"
#include "LitMaterial.h"


namespace GLRenderer
{
	LitMaterial::LitMaterial(const LitShader& shader, const math::float4& color)
		: shader(shader),
		  color(color)
	{
	}

	void LitMaterial::draw(const ::Geometry* geometry) const
	{
		shader.draw(geometry, color);
	}

	void LitMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
