


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../Geometry.h"
#include "../Renderer.h"

#include "ColoredMaterial.h"


namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace FreePipe
{
	ColoredMaterial::ColoredMaterial(Renderer& renderer, const math::float4& color, CUmodule module)
		: module(module),
		  renderer(renderer),
		  color(color)
	{
		size_t constant_size;

		succeed(cuModuleGetGlobal(&color_data, &constant_size, module, "c_SimpleColorData"));
		
		succeed(cuModuleGetFunction(&kernel_fragment_processing, module, "runFragmentStageColored"));
	}


	void ColoredMaterial::draw(const ::Geometry* geometry) const
	{
		// set light stuff


		succeed(cuMemcpyHtoD(color_data, &color, sizeof(math::float3)));

		geometry->draw();
		
		//renderer.runVertexShader(static_cast<const IndexedGeometry*>(geometry)->getNumPatches(), static_cast<const IndexedGeometry*>(geometry)->getNumVertices(), 3 * static_cast<const IndexedGeometry*>(geometry)->getNumTriangles(), false, false);

		// TOOO: readback num triangles
		unsigned int block_size = 512;
		unsigned int numTriangles = static_cast<unsigned int>(static_cast<const IndexedGeometry*>(geometry)->getNumTriangles());
		unsigned int num_blocks = divup(numTriangles, block_size);

		void* params[] = {
			&numTriangles
		};
		renderer.beginTimingRasterization();
		succeed(cuLaunchKernel(kernel_fragment_processing, num_blocks, 1, 1, block_size, 1, 1, 0, 0, params, nullptr));
		renderer.endTimingRasterization();
		//succeed(cuCtxSynchronize());
	}

	void ColoredMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
