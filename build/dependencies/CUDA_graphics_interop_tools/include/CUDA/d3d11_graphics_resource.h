


#ifndef INCLUDED_CUDA_D3D11_GRAPHICS_RESOURCE
#define INCLUDED_CUDA_D3D11_GRAPHICS_RESOURCE

#pragma once

#include <d3d11.h>

#include <cuda.h>
#include <cudaD3D11.h>

#include "graphics_resource.h"


namespace CU
{
	namespace graphics
	{
		unique_resource registerD3D11Resource(ID3D11Resource* resource, unsigned int flags = CU_GRAPHICS_REGISTER_FLAGS_NONE);
	}
}

#endif  // INCLUDED_CUDA_D3D11_GRAPHICS_RESOURCE
