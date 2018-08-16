


#include "TexturedLitMaterial.h"
#include "../shaders/fragment_phong.h"
#include "../Texture.h"
#include "../Geometry.h"
#include "../Renderer.h"

#include <CUDA/module.h>
#include <CUDA/error.h>

namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace FreePipe
{
	TexturedLitMaterial::TexturedLitMaterial(Renderer& renderer, Texture& tex, const math::float4& color, CUmodule module)
		: renderer(renderer), tex(tex), module(module), color(color)
	{
		size_t constant_size;

		succeed(cuModuleGetGlobal(&phong_data, &constant_size, module, "c_PhongData"));
		succeed(cuModuleGetGlobal(&light_data, &constant_size, module, "c_LightPos"));

		succeed(cuModuleGetFunction(&kernel_fragment_processing, module, "runFragmentStageTexPhong"));
		succeed(cuModuleGetTexRef(&texref, module, "t_phongTex"));
	}

	void TexturedLitMaterial::draw(const ::Geometry* geometry) const
	{
		// tex setup
		succeed(cuTexRefSetArray(texref, tex.getArray(), CU_TRSA_OVERRIDE_FORMAT));
		succeed(cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES));
		succeed(cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_WRAP));

		// set light stuff
		math::float3 lightpos = renderer.getLightPos();
		succeed(cuMemcpyHtoD(light_data, &lightpos, sizeof(math::float3)));

		// mat stuff
		Shaders::PhongData mat;
		mat.materialDiffuseColor = color.xyz();
		mat.materialSpecularColor = math::float3(1.0f);
		mat.diffuseAlpha = 1.0f;
		mat.specularAlpha = 0.2f;
		mat.specularExp = 9.28f;
		mat.lightColor = math::float3(1.0f);

		succeed(cuMemcpyHtoD(phong_data, &mat, sizeof(Shaders::PhongData)));

		geometry->draw();

		//renderer.runVertexShader(static_cast<const IndexedGeometry*>(geometry)->getNumPatches(), static_cast<const IndexedGeometry*>(geometry)->getNumVertices(), 3 * static_cast<const IndexedGeometry*>(geometry)->getNumTriangles(), true, true);

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


		geometry->draw();
	}

	void TexturedLitMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
