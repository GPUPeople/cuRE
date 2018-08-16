


#ifndef INCLUDED_CURE_PIPELINE_GEOMETRY_STAGE
#define INCLUDED_CURE_PIPELINE_GEOMETRY_STAGE

#pragma once

#include <CUDA/module.h>


namespace cuRE
{
	class PipelineModule;

	class GeometryStage
	{
	private:
		CUdeviceptr vertex_buffer;
		CUdeviceptr index_buffer;
		CUdeviceptr num_indices;

		CUdeviceptr index_counter;

	public:
		GeometryStage(const GeometryStage&) = delete;
		GeometryStage& operator =(const GeometryStage&) = delete;

		GeometryStage(const PipelineModule& module);

		void setVertexBuffer(CUdeviceptr vertices, size_t num_vertices);
		void setIndexBuffer(CUdeviceptr indices, size_t num_indices);
	};
}

#endif  // INCLUDED_CURE_PIPELINE_GEOMETRY_STAGE
