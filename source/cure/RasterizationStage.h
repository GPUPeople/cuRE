


#ifndef INCLUDED_CURE_PIPELINE_RASTERIZATION_STAGE
#define INCLUDED_CURE_PIPELINE_RASTERIZATION_STAGE

#pragma once

#include <CUDA/module.h>


namespace cuRE
{
	class PipelineModule;

	class RasterizationStage
	{
	private:
		CUdeviceptr viewport;
		CUdeviceptr pixel_scale;

	public:
		RasterizationStage(const RasterizationStage&) = delete;
		RasterizationStage& operator =(const RasterizationStage&) = delete;

		RasterizationStage(const PipelineModule& module);

		void setViewport(float x, float y, float width, float height);
	};
}

#endif  // INCLUDED_CURE_PIPELINE_RASTERIZATION_STAGE
