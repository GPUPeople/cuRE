


#include <algorithm>

#include <CUDA/error.h>

#include "utils.h"

#include "pipeline/config.h"

#include "PipelineModule.h"
#include "RasterizationStage.h"


namespace
{
	void initRasterizationStage(const cuRE::PipelineModule& module)
	{
		CUfunction init_kernel = module.getFunction("initRasterizationStage");

		succeed(cuLaunchKernel(init_kernel, divup(std::max(RASTERIZER_QUEUE_SIZE, TRIANGLE_BUFFER_SIZE), 1024U), 1U, 1U, 1024U, 1U, 1U, 0U, 0, nullptr, nullptr));
		succeed(cuCtxSynchronize());
	}
}

namespace cuRE
{
	RasterizationStage::RasterizationStage(const PipelineModule& module)
		: viewport(module.getGlobal("viewport")),
		  pixel_scale(module.getGlobal("pixel_scale"))
	{
		initRasterizationStage(module);
	}

	void RasterizationStage::setViewport(float x, float y, float width, float height)
	{
		float vp[] = { x, y, x + width, y + height };
		succeed(cuMemcpyHtoD(viewport, vp, sizeof(vp)));

		float ps[] = { 2.0f / width, 2.0f / height, -1.0f - x * 2.0f / width + 1.0f / width, -1.0f - y * 2.0f / height + 1.0f / height };
		succeed(cuMemcpyHtoD(pixel_scale, ps, sizeof(ps)));
	}
}
