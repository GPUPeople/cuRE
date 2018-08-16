


#ifndef INCLUDED_CUDA_GRAPHICS_RESOURCE
#define INCLUDED_CUDA_GRAPHICS_RESOURCE

#pragma once

#include <cuda.h>

#include <CUDA/error.h>
#include <CUDA/unique_handle.h>


namespace CU
{
	namespace graphics
	{
		struct GraphicsUnregisterResourceDeleter
		{
			void operator ()(CUgraphicsResource resource) const
			{
				cuGraphicsUnregisterResource(resource);
			}
		};
		
		typedef unique_handle<CUgraphicsResource, 0, GraphicsUnregisterResourceDeleter> unique_resource;
		
		
		class map_resources
		{
		private:
			CUgraphicsResource* resources;
			unsigned int count;
			CUstream stream;
			
		public:
			map_resources(const map_resources&) = delete;
			map_resources& operator =(const map_resources&) = delete;
			
			map_resources(CUgraphicsResource* resources, unsigned int count, CUstream stream = 0)
				: resources(resources),
				  count(count),
				  stream(stream)
			{
				succeed(cuGraphicsMapResources(count, resources, stream));
			}
			
			template <int N>
			map_resources(CUgraphicsResource (&resources)[N], CUstream stream = 0)
				: resources(resources),
				  count(N),
				  stream(stream)
			{
				succeed(cuGraphicsMapResources(count, resources, stream));
			}
			
			~map_resources()
			{
				cuGraphicsUnmapResources(count, resources, stream);
			}
		};
		
		
		struct mapped_resource_pointer
		{
			CUdeviceptr ptr;
			size_t size;
			
			mapped_resource_pointer(CUdeviceptr ptr, size_t size)
				: ptr(ptr),
				  size(size)
			{
			}
		};
		
		mapped_resource_pointer getMappedPointer(CUgraphicsResource resource);
		
		CUarray getMappedArray(CUgraphicsResource resource, unsigned int array_index, unsigned int mip_level);
		CUmipmappedArray getMappedMipmappedArray(CUgraphicsResource resource);
	}
}

#endif  // INCLUDED_CUDA_GRAPHICS_RESOURCE
