


#ifndef INCLUDED_CUDA_SURFACE
#define INCLUDED_CUDA_SURFACE

#pragma once

#include <cstddef>

#include <cuda.h>

#include <CUDA/unique_handle.h>


namespace CU
{
	struct SurfObjectDestroyDeleter
	{
		void operator ()(CUsurfObject surface) const
		{
			cuSurfObjectDestroy(surface);
		}
	};
	
	using unique_surface = unique_handle<CUsurfObject, 0ULL, SurfObjectDestroyDeleter>;
	
	unique_surface createSurfaceObject(const CUDA_RESOURCE_DESC& desc);
	unique_surface createSurfaceObject(CUarray array);
	unique_surface createSurfaceObject(CUmipmappedArray mipmapped_array);
	unique_surface createSurfaceObject(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t size);
	unique_surface createSurfaceObject(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t width, std::size_t height, std::size_t pitch);
}

#endif  // INCLUDED_CUDA_SURFACE
