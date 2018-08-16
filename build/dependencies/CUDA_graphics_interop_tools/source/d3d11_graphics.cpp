


#include <CUDA/error.h>

#include "d3d11_graphics_resource.h"


namespace CU
{
	namespace graphics
	{
		unique_resource registerD3D11Resource(ID3D11Resource* resource, unsigned int flags)
		{
			CUgraphicsResource cuda_resource;
			succeed(cuGraphicsD3D11RegisterResource(&cuda_resource, resource, flags));
			return unique_resource(cuda_resource);
		}
	}
}
