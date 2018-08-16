


#include <algorithm>

#include <CUDA/error.h>

#include "config.h"
#include "VertexShader.h"
#include "Renderer.h"


namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace FreePipe
{
	VertexShader::VertexShader(Renderer& renderer, CUmodule module)
		: renderer(renderer)
	{
//#if CLIPSPACE_GEOMETRY
//		succeed(cuModuleGetFunction(&kernel_geometry_processing_clipspace, module, "runGeometryStageClipSpace"));
//#else
//		succeed(cuModuleGetFunction(&kernel_geometry_processing, module, "runGeometryStageSimpleVertex"));
//		succeed(cuModuleGetFunction(&kernel_geometry_processing_tex, module, "runGeometryStageSimpleVertexTex"));
//		succeed(cuModuleGetFunction(&kernel_geometry_processing_light, module, "runGeometryStageSimpleVertexLight"));
//		succeed(cuModuleGetFunction(&kernel_geometry_processing_lighttex, module, "runGeometryStageSimpleVertexLightTex"));
//#endif
	}

	void VertexShader::run(unsigned int num_patches, unsigned int num_vertices, unsigned int num_indices, bool light, bool tex)
	{
//		unsigned int block_size;
//		unsigned int num_blocks;
//		unsigned int smem = 0;
//
//		// run vertex stage
//		switch (GEOMETRY_PROCESSING)
//		{
//		case GPM_ALLVERTICES:
//			block_size = GPM_ALLVERTICES_THREADS;
//			num_blocks = divup(std::min(num_vertices, num_indices), block_size);
//			break;
//		case GPM_ALLINDICES:
//			block_size = 384;
//			num_blocks = num_patches;
//			break;
//		case GPM_SORTING:
//			block_size = GPM_PATCH_MAX_VERTICES;
//			num_blocks = num_patches;
//			smem = (GPM_PATCH_MAX_INDICES + 1) * 8u;
//			break;
//		case GPM_HASHING:
//		case GPM_HASHING_COLLABORATIVE:
//			block_size = GPM_PATCH_MAX_VERTICES;
//			num_blocks = num_patches;
//			smem = GPM_PATCH_MAX_VERTICES * 4u;
//			break;
//		case GPM_WARP_VOTING:
//			block_size = KERNEL_THREADS;
//			num_blocks = divup(num_patches, (block_size / 32u));
//			break;
//		case GPM_WARP_VOTING_NOPREPATCHING:
//			block_size = KERNEL_THREADS;
//			num_blocks = divup(divup(num_indices, GPM_WARP_NOPREPATCHING_INDICES), (block_size / 32u));
//			smem = (block_size / 32u)*GPM_WARP_NOPREPATCHING_INDICES * 4;
//			break;
//		}
//
//		void* params[] = {
//			&num_patches,
//			&num_vertices,
//			&num_indices
//		};
//
//		renderer.beginTimingGeometry();
//
//#if CLIPSPACE_GEOMETRY
//        succeed(cuLaunchKernel(kernel_geometry_processing_clipspace, num_blocks, 1, 1, block_size, 1, 1, smem, 0, params, nullptr));
//#else
//		if (light && tex)
//			succeed(cuLaunchKernel(kernel_geometry_processing_lighttex, num_blocks, 1, 1, block_size, 1, 1, smem, 0, params, nullptr));
//		else if (light)
//			succeed(cuLaunchKernel(kernel_geometry_processing_light, num_blocks, 1, 1, block_size, 1, 1, smem, 0, params, nullptr));
//		else if (tex)
//			succeed(cuLaunchKernel(kernel_geometry_processing_tex, num_blocks, 1, 1, block_size, 1, 1, smem, 0, params, nullptr));
//		else
//			succeed(cuLaunchKernel(kernel_geometry_processing, num_blocks, 1, 1, block_size, 1, 1, smem, 0, params, nullptr));
//#endif
//		renderer.endTimingGeometry();
//		//succeed(cuCtxSynchronize());
		renderer.beginTimingGeometry();
		renderer.endTimingGeometry();
	}
}
