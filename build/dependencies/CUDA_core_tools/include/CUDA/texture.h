


#ifndef INCLUDED_CUDA_TEXTURE
#define INCLUDED_CUDA_TEXTURE

#pragma once

#include <limits>
#include <cstddef>

#include <cuda.h>

#include <CUDA/unique_handle.h>


namespace CU
{
	CUDA_TEXTURE_DESC initTextureDescriptor(CUaddress_mode address_mode_u = CU_TR_ADDRESS_MODE_CLAMP, CUaddress_mode address_mode_v = CU_TR_ADDRESS_MODE_CLAMP, CUaddress_mode address_mode_w = CU_TR_ADDRESS_MODE_CLAMP, CUfilter_mode filter_mode = CU_TR_FILTER_MODE_POINT, unsigned int flags = 0U, unsigned int max_anisotropy = 1U, CUfilter_mode mipmap_filter_mode = CU_TR_FILTER_MODE_POINT, float level_bias = 0.0f, float min_level_clamp = -std::numeric_limits<float>::max(), float max_level_clamp = std::numeric_limits<float>::max());
	CUDA_TEXTURE_DESC initTextureDescriptor(CUaddress_mode address_mode_u, CUaddress_mode address_mode_v, CUfilter_mode filter_mode, unsigned int flags = 0U, unsigned int max_anisotropy = 1U, CUfilter_mode mipmap_filter_mode = CU_TR_FILTER_MODE_POINT, float level_bias = 0.0f, float min_level_clamp = -std::numeric_limits<float>::max(), float max_level_clamp = std::numeric_limits<float>::max());
	CUDA_TEXTURE_DESC initTextureDescriptor(CUaddress_mode address_mode_u, CUfilter_mode filter_mode, unsigned int flags = 0U, unsigned int max_anisotropy = 1U, CUfilter_mode mipmap_filter_mode = CU_TR_FILTER_MODE_POINT, float level_bias = 0.0f, float min_level_clamp = -std::numeric_limits<float>::max(), float max_level_clamp = std::numeric_limits<float>::max());

	void initTextureDescriptor(CUDA_TEXTURE_DESC& desc, CUaddress_mode address_mode_u = CU_TR_ADDRESS_MODE_CLAMP, CUaddress_mode address_mode_v = CU_TR_ADDRESS_MODE_CLAMP, CUaddress_mode address_mode_w = CU_TR_ADDRESS_MODE_CLAMP, CUfilter_mode filter_mode = CU_TR_FILTER_MODE_POINT, unsigned int flags = 0U, unsigned int max_anisotropy = 1U, CUfilter_mode mipmap_filter_mode = CU_TR_FILTER_MODE_POINT, float level_bias = 0.0f, float min_level_clamp = -std::numeric_limits<float>::max(), float max_level_clamp = std::numeric_limits<float>::max());
	void initTextureDescriptor(CUDA_TEXTURE_DESC& desc, CUaddress_mode address_mode_u, CUaddress_mode address_mode_v, CUfilter_mode filter_mode, unsigned int flags = 0U, unsigned int max_anisotropy = 1U, CUfilter_mode mipmap_filter_mode = CU_TR_FILTER_MODE_POINT, float level_bias = 0.0f, float min_level_clamp = -std::numeric_limits<float>::max(), float max_level_clamp = std::numeric_limits<float>::max());
	void initTextureDescriptor(CUDA_TEXTURE_DESC& desc, CUaddress_mode address_mode_u, CUfilter_mode filter_mode, unsigned int flags = 0U, unsigned int max_anisotropy = 1U, CUfilter_mode mipmap_filter_mode = CU_TR_FILTER_MODE_POINT, float level_bias = 0.0f, float min_level_clamp = -std::numeric_limits<float>::max(), float max_level_clamp = std::numeric_limits<float>::max());


	struct TexObjectDestroyDeleter
	{
		void operator ()(CUtexObject texture) const
		{
			cuTexObjectDestroy(texture);
		}
	};
	
	using unique_texture = unique_handle<CUtexObject, 0ULL, TexObjectDestroyDeleter>;
	
	unique_texture createTextureObject(const CUDA_RESOURCE_DESC& resource_desc, const CUDA_RESOURCE_VIEW_DESC& resource_view_desc, const CUDA_TEXTURE_DESC& texture_desc);
	unique_texture createTextureObject(const CUDA_RESOURCE_DESC& resource_desc, const CUDA_TEXTURE_DESC& texture_desc);
}

#endif  // INCLUDED_CUDA_TEXTURE
