


#include "GLRenderer/shaders/ColoredShader.h"
#include "ColoredMaterial.h"


namespace GLRenderer
{
	ColoredMaterial::ColoredMaterial(const ColoredShader& shader, const math::float4& color)
		: shader(shader),
		  color(color)
	{
	}

	void ColoredMaterial::draw(const ::Geometry* geometry) const
	{
		shader.draw(geometry, color);
	}

	void ColoredMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
