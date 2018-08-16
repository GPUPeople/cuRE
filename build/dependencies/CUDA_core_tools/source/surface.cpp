


#include <CUDA/error.h>
#include <CUDA/resource.h>
#include <CUDA/surface.h>


namespace CU
{
	unique_surface createSurfaceObject(const CUDA_RESOURCE_DESC& desc)
	{
		CUsurfObject surf;
		succeed(cuSurfObjectCreate(&surf, &desc));
		return unique_surface(surf);
	}

	unique_surface createSurfaceObject(CUarray array)
	{
		return createSurfaceObject(initResourceDescriptor(array));
	}

	unique_surface createSurfaceObject(CUmipmappedArray mipmapped_array)
	{
		return createSurfaceObject(initResourceDescriptor(mipmapped_array));
	}

	unique_surface createSurfaceObject(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t size)
	{
		return createSurfaceObject(initResourceDescriptor(memory, format, num_channels, size));
	}

	unique_surface createSurfaceObject(CUdeviceptr memory, CUarray_format format, unsigned int num_channels, std::size_t width, std::size_t height, std::size_t pitch)
	{
		return createSurfaceObject(initResourceDescriptor(memory, format, num_channels, width, height, pitch));
	}
}
