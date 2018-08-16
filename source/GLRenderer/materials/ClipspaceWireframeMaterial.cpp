


#include "GLRenderer/shaders/ClipspaceShader.h"
#include "ClipspaceMaterial.h"


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
}
