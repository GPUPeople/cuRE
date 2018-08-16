


#include <GL/error.h>

#include <ResourceImp.h>

#include "materials/TexturedMaterial.h"
#include "materials/TexturedLitMaterial.h"

#include "Renderer.h"
#include "Texture.h"


namespace GLRenderer
{
	Texture2DRGBA8::Texture2DRGBA8(Renderer& renderer, GLsizei width, GLsizei height, GLsizei levels, const std::uint32_t* data)
		: renderer(renderer)
	{
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexStorage2D(GL_TEXTURE_2D, levels, GL_SRGB8_ALPHA8, width, height);

		for (GLsizei i = 0; i < levels; ++i)
		{
			glTexSubImage2D(GL_TEXTURE_2D, i, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);

			data += width * height;

			width = std::max(width / 2, 1);
			height = std::max(height / 2, 1);
		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		GL::throw_error();
	}

	::Material* Texture2DRGBA8::createTexturedMaterial(const math::float4& color)
	{
		return renderer.createTexturedMaterial(tex, color);
	}

	::Material* Texture2DRGBA8::createTexturedLitMaterial(const math::float4& color)
	{
		return renderer.createTexturedLitMaterial(tex, color);
	}
}
