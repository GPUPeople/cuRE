


#include <CUDA/error.h>
#include <CUDA/resource.h>
#include <CUDA/texture.h>


namespace CU
{
	CUDA_TEXTURE_DESC initTextureDescriptor(CUaddress_mode address_mode_u, CUaddress_mode address_mode_v, CUaddress_mode address_mode_w, CUfilter_mode filter_mode, unsigned int flags, unsigned int max_anisotropy, CUfilter_mode mipmap_filter_mode, float level_bias, float min_level_clamp, float max_level_clamp)
	{
		return { address_mode_u, address_mode_v, address_mode_w, filter_mode, flags, max_anisotropy, mipmap_filter_mode, level_bias, min_level_clamp, max_level_clamp };
	}

	CUDA_TEXTURE_DESC initTextureDescriptor(CUaddress_mode address_mode_u, CUaddress_mode address_mode_v, CUfilter_mode filter_mode, unsigned int flags, unsigned int max_anisotropy, CUfilter_mode mipmap_filter_mode, float level_bias, float min_level_clamp, float max_level_clamp)
	{
		return initTextureDescriptor(address_mode_u, address_mode_v, CU_TR_ADDRESS_MODE_CLAMP, filter_mode, flags, max_anisotropy, mipmap_filter_mode, level_bias, min_level_clamp, max_level_clamp);
	}

	CUDA_TEXTURE_DESC initTextureDescriptor(CUaddress_mode address_mode_u, CUfilter_mode filter_mode, unsigned int flags, unsigned int max_anisotropy, CUfilter_mode mipmap_filter_mode, float level_bias, float min_level_clamp, float max_level_clamp)
	{
		return initTextureDescriptor(address_mode_u, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, filter_mode, flags, max_anisotropy, mipmap_filter_mode, level_bias, min_level_clamp, max_level_clamp);
	}

	void initTextureDescriptor(CUDA_TEXTURE_DESC& desc, CUaddress_mode address_mode_u, CUaddress_mode address_mode_v, CUaddress_mode address_mode_w, CUfilter_mode filter_mode, unsigned int flags, unsigned int max_anisotropy, CUfilter_mode mipmap_filter_mode, float level_bias, float min_level_clamp, float max_level_clamp)
	{
		desc = initTextureDescriptor(address_mode_u, address_mode_v, address_mode_w, filter_mode, flags, max_anisotropy, mipmap_filter_mode, level_bias, min_level_clamp, max_level_clamp);
	}

	void initTextureDescriptor(CUDA_TEXTURE_DESC& desc, CUaddress_mode address_mode_u, CUaddress_mode address_mode_v, CUfilter_mode filter_mode, unsigned int flags, unsigned int max_anisotropy, CUfilter_mode mipmap_filter_mode, float level_bias, float min_level_clamp, float max_level_clamp)
	{
		desc = initTextureDescriptor(address_mode_u, address_mode_v, filter_mode, flags, max_anisotropy, mipmap_filter_mode, level_bias, min_level_clamp, max_level_clamp);
	}

	void initTextureDescriptor(CUDA_TEXTURE_DESC& desc, CUaddress_mode address_mode_u, CUfilter_mode filter_mode, unsigned int flags, unsigned int max_anisotropy, CUfilter_mode mipmap_filter_mode, float level_bias, float min_level_clamp, float max_level_clamp)
	{
		desc = initTextureDescriptor(address_mode_u, filter_mode, flags, max_anisotropy, mipmap_filter_mode, level_bias, min_level_clamp, max_level_clamp);
	}


	unique_texture createTextureObject(const CUDA_RESOURCE_DESC& resource_desc, const CUDA_RESOURCE_VIEW_DESC& resource_view_desc, const CUDA_TEXTURE_DESC& texture_desc)
	{
		CUtexObject tex;
		succeed(cuTexObjectCreate(&tex, &resource_desc, &texture_desc, &resource_view_desc));
		return unique_texture(tex);
	}

	unique_texture createTextureObject(const CUDA_RESOURCE_DESC& resource_desc, const CUDA_TEXTURE_DESC& texture_desc)
	{
		CUtexObject tex;
		succeed(cuTexObjectCreate(&tex, &resource_desc, &texture_desc, nullptr));
		return unique_texture(tex);
	}
}
