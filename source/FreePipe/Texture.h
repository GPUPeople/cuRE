


#ifndef INCLUDED_FREEPIPE_TEXTURE
#define INCLUDED_FREEPIPE_TEXTURE

#include <cstdint>

#include <CUDA/memory.h>
#include <CUDA/array.h>

#include <Resource.h>


namespace FreePipe
{
	class Renderer;
	class Texture : public ::Texture
	{
	protected:
		Texture(const Texture&) = delete;
		Texture& operator =(const Texture&) = delete;

		Renderer& renderer;

		CU::unique_array tex_array;

	public:
		Texture(Renderer& renderer, size_t width, size_t height, unsigned int levels, const std::uint32_t* data);

		CUarray getArray() { return tex_array; }

		::Material* createTexturedMaterial(const math::float4& color);
		::Material* createTexturedLitMaterial(const math::float4& color);
	};
}

#endif  // INCLUDED_FREEPIPE_TEXTURE
