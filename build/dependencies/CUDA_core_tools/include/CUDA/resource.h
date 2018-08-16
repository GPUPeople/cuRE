


#ifndef INCLUDED_CUDA_RESOURCE
#define INCLUDED_CUDA_RESOURCE

#pragma once

#include <cstddef>

#include <cuda.h>


namespace CU
{
	CUDA_RESOURCE_DESC initResourceDescriptor(CUarray array, unsigned int flags = 0U);
	CUDA_RESOURCE_DESC initResourceDescriptor(CUmipmappedArray mipmapped_array, unsigned int flags = 0U);
	CUDA_RESOURCE_DESC initResourceDescriptor(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t size, unsigned int flags = 0U);
	CUDA_RESOURCE_DESC initResourceDescriptor(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t width, std::size_t height, std::size_t pitch, unsigned int flags = 0U);

	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUarray array, unsigned int flags = 0U);
	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUmipmappedArray mipmapped_array, unsigned int flags = 0U);
	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t size, unsigned int flags = 0U);
	void initResourceDescriptor(CUDA_RESOURCE_DESC& desc, CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t width, std::size_t height, std::size_t pitch, unsigned int flags = 0U);


	CUDA_RESOURCE_VIEW_DESC initResourceViewDescriptor(CUresourceViewFormat format, std::size_t width, std::size_t height = 1U, std::size_t depth = 1U, unsigned int first_level = 0U, unsigned int last_level = 0U, unsigned int first_layer = 0U, unsigned int last_layer = 0U);

	void initResourceViewDescriptor(CUDA_RESOURCE_VIEW_DESC& desc, CUresourceViewFormat format, std::size_t width, std::size_t height = 1U, std::size_t depth = 1U, unsigned int first_level = 0U, unsigned int last_level = 0U, unsigned int first_layer = 0U, unsigned int last_layer = 0U);
}

#endif  // INCLUDED_CUDA_RESOURCE
