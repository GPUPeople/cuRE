


#include <stdexcept>

#include <CUDA/error.h>
#include <CUDA/module.h>

#include "../Geometry.h"
#include "../Renderer.h"
#include "../shaders/fragment_phong.h"

#include "ClipspaceMaterial.h"


namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace FreePipe
{
	ClipspaceMaterial::ClipspaceMaterial(Renderer& renderer, CUmodule module)
	    : module(module),
	      renderer(renderer)
	{
		//size_t constant_size;
		succeed(cuModuleGetFunction(&kernel_fragment_processing, module, "runFragmentStageClipSpace"));
	}

	void ClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		geometry->draw();

		//renderer.runVertexShader(static_cast<const ClipspaceGeometry*>(geometry)->getNumPatches(), static_cast<const ClipspaceGeometry*>(geometry)->getNumVertices(), 3 * static_cast<const ClipspaceGeometry*>(geometry)->getNumTriangles(), false, false);

		// TOOO: readback num triangles
		unsigned int block_size = 512;
		unsigned int numTriangles = static_cast<unsigned int>(static_cast<const ClipspaceGeometry*>(geometry)->getNumTriangles());
		unsigned int num_blocks = divup(numTriangles, block_size);

		void* params[] = {
			&numTriangles
		};

		renderer.beginTimingRasterization();
		succeed(cuLaunchKernel(kernel_fragment_processing, num_blocks, 1, 1, block_size, 1, 1, 0, 0, params, nullptr));
		renderer.endTimingRasterization();
	}

	void ClipspaceMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
	}
}
