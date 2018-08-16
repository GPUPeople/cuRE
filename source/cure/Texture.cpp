


#include <algorithm>

#include <CUDA/error.h>

#include <ResourceImp.h>

#include "materials/TexturedMaterial.h"
#include "materials/TexturedLitMaterial.h"

#include "Texture.h"


namespace cuRE
{
	Texture::Texture(Pipeline& pipeline, size_t width, size_t height, unsigned int levels, const std::uint32_t* data)
		: tex(CU::createArray2DMipmapped(width, height, levels, CU_AD_FORMAT_UNSIGNED_INT8, 4)),
		  pipeline(pipeline)
	{
		for (unsigned int i = 0; i < levels; ++i)
		{
			CUarray array;
			succeed(cuMipmappedArrayGetLevel(&array, tex, i));

			CUDA_MEMCPY3D cpy;

			cpy.srcXInBytes = 0U;
			cpy.srcY = 0U;
			cpy.srcZ = 0U;
			cpy.srcLOD = 0U;
			cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
			cpy.srcHost = data;
			cpy.srcPitch = width * 4U;
			cpy.srcHeight = height;

			cpy.dstXInBytes = 0U;
			cpy.dstY = 0U;
			cpy.dstZ = 0U;
			cpy.dstLOD = 0U;
			cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			cpy.dstArray = array;

			cpy.WidthInBytes = width * 4U;
			cpy.Height = height;
			cpy.Depth = 1;

			succeed(cuMemcpy3D(&cpy));

			data += width * height;

			width = std::max<size_t>(width / 2, 1);
			height = std::max<size_t>(height / 2, 1);
		}
	}

	::Material* Texture::createTexturedMaterial(const math::float4& color)
	{
		return ResourceImp<TexturedMaterial>::create(pipeline, tex, color);
	}

	::Material* Texture::createTexturedLitMaterial(const math::float4& color)
	{
		return ResourceImp<TexturedLitMaterial>::create(pipeline, tex, color);
	}
}
