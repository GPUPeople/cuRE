


#include <ResourceImp.h>

#include "materials/TexturedMaterial.h"
#include "materials/TexturedLitMaterial.h"
#include "materials/LitMaterial.h"

#include "Texture.h"
#include "Renderer.h"


namespace FreePipe
{
	Texture::Texture(Renderer& renderer, size_t width, size_t height, unsigned int levels, const std::uint32_t* data) : renderer(renderer)
	{
		tex_array = CU::createArray2D(width, height, CU_AD_FORMAT_UNSIGNED_INT8, 4);
		CUDA_MEMCPY2D copyParams;
		copyParams.WidthInBytes = 4 * width;
		copyParams.Height = height;

		copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
		copyParams.srcHost = data;
		copyParams.srcPitch = 4 * width;
		copyParams.srcXInBytes = 0;
		copyParams.srcY = 0;

		copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		copyParams.dstArray = tex_array;
		copyParams.dstXInBytes = 0;
		copyParams.dstY = 0;

		succeed(cuMemcpy2D(&copyParams));
	}

	::Material* Texture::createTexturedMaterial(const math::float4& color)
	{
		return ResourceImp<TexturedMaterial>::create(renderer, *this, color, renderer.getModule());
	}

	::Material* Texture::createTexturedLitMaterial(const math::float4& color)
	{
		return ResourceImp<TexturedLitMaterial>::create(renderer, *this, color, renderer.getModule());
	}
}
