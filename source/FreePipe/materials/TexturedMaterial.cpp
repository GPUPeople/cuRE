


#include "TexturedMaterial.h"

#include "../shaders/fragment_phong.h"
#include "../Texture.h"
#include "../Geometry.h"
#include "../Renderer.h"

namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace FreePipe
{
	TexturedMaterial::TexturedMaterial(Renderer& renderer, Texture& tex, const math::float4& color, CUmodule module)
		: renderer(renderer), tex(tex), color(color), module(module)
	{
		size_t constant_size;

		succeed(cuModuleGetGlobal(&mat_color, &constant_size, module, "c_SimpleColorData"));
		succeed(cuModuleGetFunction(&kernel_fragment_processing, module, "runFragmentStageColoredTex"));
		succeed(cuModuleGetTexRef(&texref, module, "t_phongTex"));
	}

	void TexturedMaterial::draw(const ::Geometry* geometry) const
	{
		// tex setup
		succeed(cuTexRefSetArray(texref, tex.getArray(), CU_TRSA_OVERRIDE_FORMAT));
		succeed(cuTexRefSetFilterMode(texref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetFlags(texref, CU_TRSF_NORMALIZED_COORDINATES));
		succeed(cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_WRAP));

		// mat color
		succeed(cuMemcpyHtoD(mat_color, &color.x, sizeof(math::float3)));

		geometry->draw();

		//renderer.runVertexShader(static_cast<const IndexedGeometry*>(geometry)->getNumPatches(), static_cast<const IndexedGeometry*>(geometry)->getNumVertices(), 3 * static_cast<const IndexedGeometry*>(geometry)->getNumTriangles(), false, true);

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

	void TexturedMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
	}
}
