


#include <memory>

#include <core/utils/memory>

#include <CUDA/error.h>

#include "../Pipeline.h"
#include "OceanMaterial.h"


namespace cuRE
{
	OceanMaterial::OceanMaterial(Pipeline& pipeline, const float* img_data, size_t width, size_t height, const std::uint32_t* normal_data, size_t n_width, size_t n_height, unsigned int n_levels)
		: pipeline(pipeline)
	{
		auto col_buffer = core::make_unique_default<float[]>(width * height * 4);

		for (unsigned int i = 0; i < width * height; i++)
		{
			col_buffer[i * 4 + 0] = img_data[i * 3 + 0];
			col_buffer[i * 4 + 1] = img_data[i * 3 + 1];
			col_buffer[i * 4 + 2] = img_data[i * 3 + 2];
			col_buffer[i * 4 + 3] = 1.0f;
		}

		{
			color_array = CU::createArray2D(width, height, CU_AD_FORMAT_FLOAT, 4);

			CUDA_MEMCPY2D cpy_info;
			cpy_info.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			cpy_info.dstArray = color_array;
			cpy_info.dstXInBytes = 0;
			cpy_info.dstY = 0;

			cpy_info.srcMemoryType = CU_MEMORYTYPE_HOST;
			cpy_info.srcHost = col_buffer.get();
			cpy_info.srcPitch = width * sizeof(float) * 4;
			cpy_info.srcXInBytes = 0;
			cpy_info.srcY = 0;

			cpy_info.WidthInBytes = width * sizeof(float) * 4;
			cpy_info.Height = height;
			succeed(cuMemcpy2D(&cpy_info));
		}

		{
			normal_array = CU::createArray2DMipmapped(n_width, n_height, n_levels, CU_AD_FORMAT_UNSIGNED_INT8, 4);

			for (unsigned int i = 0; i < n_levels; ++i)
			{
				CUarray array;
				succeed(cuMipmappedArrayGetLevel(&array, normal_array, i));

				CUDA_MEMCPY3D cpy;

				cpy.srcXInBytes = 0U;
				cpy.srcY = 0U;
				cpy.srcZ = 0U;
				cpy.srcLOD = 0U;
				cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
				cpy.srcHost = normal_data;
				cpy.srcPitch = n_width * 4U;
				cpy.srcHeight = n_height;

				cpy.dstXInBytes = 0U;
				cpy.dstY = 0U;
				cpy.dstZ = 0U;
				cpy.dstLOD = 0U;
				cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
				cpy.dstArray = array;

				cpy.WidthInBytes = n_width * 4U;
				cpy.Height = n_height;
				cpy.Depth = 1;

				succeed(cuMemcpy3D(&cpy));

				normal_data += n_width * n_height;

				n_width = std::max<size_t>(n_width / 2, 1);
				n_height = std::max<size_t>(n_height / 2, 1);
			}
		}
	}

	void OceanMaterial::draw(const ::Geometry* geometry) const
	{
		pipeline.setTextureF(color_array);
		pipeline.setTexture(normal_array, std::numeric_limits<float>::max());
		geometry->draw();
	}

	void OceanMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		pipeline.setTextureF(color_array);
		pipeline.setTexture(normal_array, std::numeric_limits<float>::max());
		geometry->draw(start, num_indices);
	}
}
