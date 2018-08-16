


#ifndef INCLUDED_GLRENDERER_TEXTURE
#define INCLUDED_GLRENDERER_TEXTURE

#include <cstdint>

#include <GL/texture.h>

#include <Resource.h>


namespace GLRenderer
{
	class Renderer;

	class Texture2DRGBA8 : public ::Texture
	{
	protected:
		Texture2DRGBA8(const Texture2DRGBA8&) = delete;
		Texture2DRGBA8& operator =(const Texture2DRGBA8&) = delete;

		Renderer& renderer;

		GL::Texture tex;

	public:
		Texture2DRGBA8(Renderer& renderer, GLsizei width, GLsizei height, GLsizei levels, const std::uint32_t* data);

		::Material* createTexturedMaterial(const math::float4& color);
		::Material* createTexturedLitMaterial(const math::float4& color);
	};
}

#endif  // INCLUDED_GLRENDERER_TEXTURE
