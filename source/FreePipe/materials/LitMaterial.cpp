


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../shaders/fragment_phong.h"
#include "../Geometry.h"
#include "../Renderer.h"

#include "LitMaterial.h"


namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace FreePipe
{
	LitMaterial::LitMaterial(Renderer& renderer, const math::float4& color, CUmodule module)
		: module(module),
		  renderer(renderer),
		  color(color)
	{
		size_t constant_size;

		succeed(cuModuleGetGlobal(&phong_data, &constant_size, module, "c_PhongData"));
		succeed(cuModuleGetGlobal(&light_data, &constant_size, module, "c_LightPos"));
		
		succeed(cuModuleGetFunction(&kernel_fragment_processing, module, "runFragmentStagePhong"));
	}


	void LitMaterial::draw(const ::Geometry* geometry) const
	{
		// set light stuff
		math::float3 lightpos = renderer.getLightPos();
		succeed(cuMemcpyHtoD(light_data, &lightpos, sizeof(math::float3)));


		Shaders::PhongData mat;
		mat.materialDiffuseColor = color.xyz();
		mat.materialSpecularColor = math::float3(1.0f);
		mat.diffuseAlpha = 1.0f;
		mat.specularAlpha = 0.2f;
		mat.specularExp = 9.28f;
		mat.lightColor = math::float3(1.0f);

		succeed(cuMemcpyHtoD(phong_data, &mat, sizeof(Shaders::PhongData)));

		geometry->draw();
		
		//renderer.runVertexShader(static_cast<const IndexedGeometry*>(geometry)->getNumPatches(), static_cast<const IndexedGeometry*>(geometry)->getNumVertices(), 3 * static_cast<const IndexedGeometry*>(geometry)->getNumTriangles(), true, false);

		



		//TOOO: readback num triangles
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

	void LitMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
