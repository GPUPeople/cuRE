


#include <CUDA/error.h>
#include <CUDA/module.h>

#include "utils.h"

#include "PipelineModule.h"
#include "GeometryStage.h"


namespace cuRE
{
	GeometryStage::GeometryStage(const PipelineModule& module)
		: vertex_buffer(module.getGlobal("vertex_buffer")),
		  index_buffer(module.getGlobal("index_buffer")),
		  num_indices(module.getGlobal("num_indices")),
		  index_counter(module.getGlobal("index_counter"))
	{
	}

	void GeometryStage::setVertexBuffer(CUdeviceptr vertices, size_t num_vertices)
	{
		succeed(cuMemcpyHtoD(this->vertex_buffer, &vertices, sizeof(CUdeviceptr)));
	}

	void GeometryStage::setIndexBuffer(CUdeviceptr indices, size_t num_indices)
	{
		succeed(cuMemcpyHtoD(this->index_buffer, &indices, sizeof(CUdeviceptr)));
		succeed(cuMemcpyHtoD(this->num_indices, &num_indices, 4U));

		unsigned int index_counter = 0;
		succeed(cuMemcpyHtoD(this->index_counter, &index_counter, 4U));
	}
}
