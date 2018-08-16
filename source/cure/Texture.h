


#ifndef INCLUDED_CURE_TEXTURE
#define INCLUDED_CURE_TEXTURE

#include <cstdint>

#include <CUDA/array.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class Texture : public ::Texture
	{
	protected:
		Texture(const Texture&) = delete;
		Texture& operator =(const Texture&) = delete;

		CU::unique_mipmapped_array tex;

		Pipeline& pipeline;

	public:
		Texture(Pipeline& pipeline, size_t width, size_t height, unsigned int levels, const std::uint32_t* data);

		::Material* createTexturedMaterial(const math::float4& color);
		::Material* createTexturedLitMaterial(const math::float4& color);
	};
}

#endif  // INCLUDED_CURE_TEXTURE
