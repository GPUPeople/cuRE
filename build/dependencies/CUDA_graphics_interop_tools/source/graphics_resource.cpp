


#include <CUDA/error.h>

#include "graphics_resource.h"


namespace CU
{
	namespace graphics
	{
		mapped_resource_pointer getMappedPointer(CUgraphicsResource resource)
		{
			CUdeviceptr ptr;
			size_t size;
			succeed(cuGraphicsResourceGetMappedPointer(&ptr, &size, resource));
			return mapped_resource_pointer(ptr, size);
		}
		
		CUarray getMappedArray(CUgraphicsResource resource, unsigned int array_index, unsigned int mip_level)
		{
			CUarray array;
			succeed(cuGraphicsSubResourceGetMappedArray(&array, resource, array_index, mip_level));
			return array;
		}
		
		CUmipmappedArray getMappedMipmappedArray(CUgraphicsResource resource)
		{
			CUmipmappedArray array;
			succeed(cuGraphicsResourceGetMappedMipmappedArray(&array, resource));
			return array;
		}
	}
}
