


#include <CUDA/resource.h>


namespace CU
{
	CUDA_RESOURCE_DESC initResourceDescriptor(CUarray array, unsigned int flags)
	{
		CUDA_RESOURCE_DESC desc;
		desc.resType = CU_RESOURCE_TYPE_ARRAY;
		desc.res.array.hArray = array;
		desc.flags = flags;

		return desc;
	}

	CUDA_RESOURCE_DESC initResourceDescriptor(CUmipmappedArray mipmapped_array, unsigned int flags)
	{
		CUDA_RESOURCE_DESC desc;
		desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
		desc.res.mipmap.hMipmappedArray = mipmapped_array;
		desc.flags = flags;

		return desc;
	}

	CUDA_RESOURCE_DESC initResourceDescriptor(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t size, unsigned int flags)
	{
		CUDA_RESOURCE_DESC desc;
		desc.resType = CU_RESOURCE_TYPE_LINEAR;
		desc.res.linear.devPtr = memory;
		desc.res.linear.format = format;
		desc.res.linear.numChannels = num_channels;
		desc.res.linear.sizeInBytes = size;
		desc.flags = flags;

		return desc;
	}

	CUDA_RESOURCE_DESC initResourceDescriptor(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t width, std::size_t height, std::size_t pitch, unsigned int flags)
	{
		CUDA_RESOURCE_DESC desc;
		desc.resType = CU_RESOURCE_TYPE_PITCH2D;
		desc.res.pitch2D.devPtr = memory;
		desc.res.pitch2D.format = format;
		desc.res.pitch2D.numChannels = num_channels;
		desc.res.pitch2D.width = width;
		desc.res.pitch2D.height = height;
		desc.res.pitch2D.pitchInBytes = pitch;
		desc.flags = flags;

		return desc;
	}


	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUarray array, unsigned int flags)
	{
		desc = initResourceDescriptor(array, flags);
	}

	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUmipmappedArray mipmapped_array, unsigned int flags)
	{
		desc = initResourceDescriptor(mipmapped_array, flags);
	}

	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t size, unsigned int flags)
	{
		desc = initResourceDescriptor(memory, format, num_channels, size, flags);
	}

	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t width, std::size_t height, std::size_t pitch, unsigned int flags)
	{
		desc = initResourceDescriptor(memory, format, num_channels, width, height, pitch, flags);
	}


	CUDA_RESOURCE_VIEW_DESC initResourceViewDescriptor(CUresourceViewFormat format, std::size_t width, std::size_t height, std::size_t depth, unsigned int first_level, unsigned int last_level, unsigned int first_layer, unsigned int last_layer)
	{
		return { format, width, height, depth, first_level, last_level, first_layer, last_layer };
	}

	void initResourceViewDescriptor(CUDA_RESOURCE_VIEW_DESC& desc, CUresourceViewFormat format, std::size_t width, std::size_t height, std::size_t depth, unsigned int first_level, unsigned int last_level, unsigned int first_layer, unsigned int last_layer)
	{
		desc = initResourceViewDescriptor(format, width, height, depth, first_level, last_level, first_layer, last_layer);
	}
}
