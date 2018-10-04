


#include <iostream>

#include <CUDA/error.h>
#include <CUDA/memory.h>

#include "utils.h"

#include "PipelineModule.h"
#include "Pipeline.h"


namespace
{
	void checkQueues(CUmodule module)
	{
		CUfunction check_kernel = CU::getFunction(module, "checkQueues");

		succeed(cuLaunchKernel(check_kernel, 32U, 1U, 1U, 1024U, 1U, 1U, 0U, 0, nullptr, nullptr));
		succeed(cuCtxSynchronize());
	}
}

namespace cuRE
{
	Pipeline::Pipeline(const PipelineModule& module)
		: geometry_stage(module),
		  rasterization_stage(module),
		  framebuffer(module),
		  shader_state(module),
		  uniform(module.getGlobal("uniform")),
		  tex_ref(module.getTextureReference("tex")),
		  texf_ref(module.getTextureReference("texf")),
		  stipple(module.getGlobal("stippling_mask")),
		  simple_shading_tris_kernel(module.findPipelineKernel("simple_shading_tris")),
		  simple_shading_quads_kernel(module.findPipelineKernel("simple_shading_quads")),
		  textured_shading_tris_kernel(module.findPipelineKernel("textured_shading_tris")),
		  textured_shading_quads_kernel(module.findPipelineKernel("textured_shading_quads")),
		  vertex_heavy_tris_kernel(module.findPipelineKernel("vertex_heavy_tris")),
		  fragment_heavy_tris_kernel(module.findPipelineKernel("fragment_heavy_tris")),
		  clipspace_shading_kernel(module.findPipelineKernel("clipspace_shading")),
		  vertex_heavy_clipspace_shading_kernel(module.findPipelineKernel("vertex_heavy_clipspace_shading")),
		  fragment_heavy_clipspace_shading_kernel(module.findPipelineKernel("fragment_heavy_clipspace_shading")),
		  eyecandy_shading_kernel(module.findPipelineKernel("eyecandy_shading")),
		  vertex_heavy_eyecandy_shading_kernel(module.findPipelineKernel("vertex_heavy_eyecandy_shading")),
		  fragment_heavy_eyecandy_shading_kernel(module.findPipelineKernel("fragment_heavy_eyecandy_shading")),
		  ocean_adaptive_kernel(module.findPipelineKernel("ocean_adaptive")),
		  ocean_static_kernel(module.findPipelineKernel("ocean_normal")),
		  ocean_adaptive_wire_kernel(module.findPipelineKernel("ocean_adaptive_wire")),
		  ocean_static_wire_kernel(module.findPipelineKernel("ocean_normal_wire")),
		  blend_demo_kernel(module.findPipelineKernel("blend_demo")),
		  iso_blend_demo_kernel(module.findPipelineKernel("iso_blend_demo")),
		  iso_stipple_kernel(module.findPipelineKernel("iso_stipple_demo")),
		  glyph_demo_kernel(module.findPipelineKernel("glyph_demo")),
		  checkerboard_fragment_kernel(module.findPipelineKernel("checkerboard_fragment_demo")),
		  checkerboard_kernel(module.findPipelineKernel("checkerboard_demo")),
		  checkerboard_quad_fragment_kernel(module.findPipelineKernel("checkerboard_quad_fragment_demo")),
		  checkerboard_quad_kernel(module.findPipelineKernel("checkerboard_quad_demo")),
		  begin_draw_event(CU::createEvent()),
		  end_draw_event(CU::createEvent()),
		  instrumentation(module)
	{
		std::cout << "using ";
		simple_shading_tris_kernel.printInfo(std::cout);
		std::cout << '\n';

		std::cout << "using ";
		simple_shading_quads_kernel.printInfo(std::cout);
		std::cout << '\n';
	}

	void Pipeline::attachColorBuffer(CUarray color_buffer, int width, int height)
	{
		framebuffer.attachColorBuffer(color_buffer, width, height);
	}

	void Pipeline::attachDepthBuffer(CUarray depth_buffer, int width, int height)
	{
		framebuffer.attachDepthBuffer(depth_buffer, width, height);
	}

	void Pipeline::clearColorBuffer(float r, float g, float b, float a)
	{
		framebuffer.clearColorBuffer(r, g, b, a);
	}

	void Pipeline::clearColorBufferCheckers(uint32_t a, uint32_t b, unsigned int s)
	{
		framebuffer.clearColorBufferCheckers(a, b, s);
	}

	void Pipeline::clearDepthBuffer(float depth)
	{
		framebuffer.clearDepthBuffer(depth);
	}

	void Pipeline::setViewport(float x, float y, float width, float height)
	{
		rasterization_stage.setViewport(x, y, width, height);
	}

	void Pipeline::setUniformi(int index, int v)
	{
		succeed(cuMemcpyHtoD(uniform + 4U * index, &v, 4U));
	}

	void Pipeline::setUniformf(int index, float v)
	{
		succeed(cuMemcpyHtoD(uniform + 4U * index, &v, 4U));
	}

	void Pipeline::setCamera(const Camera::UniformBuffer& camera)
	{
		shader_state.setCamera(camera);
	}

	void Pipeline::setTexture(CUarray tex)
	{
		succeed(cuTexRefSetArray(tex_ref, tex, CU_TRSA_OVERRIDE_FORMAT));
		succeed(cuTexRefSetFlags(tex_ref, CU_TRSF_NORMALIZED_COORDINATES));
		succeed(cuTexRefSetFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetMipmapFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetAddressMode(tex_ref, 0, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetAddressMode(tex_ref, 1, CU_TR_ADDRESS_MODE_WRAP));
	}

	void Pipeline::setTextureSRGB(CUarray tex)
	{
		succeed(cuTexRefSetArray(tex_ref, tex, CU_TRSA_OVERRIDE_FORMAT));
		succeed(cuTexRefSetFlags(tex_ref, CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB));
		succeed(cuTexRefSetFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetMipmapFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetAddressMode(tex_ref, 0, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetAddressMode(tex_ref, 1, CU_TR_ADDRESS_MODE_WRAP));
	}

	void Pipeline::setTextureF(CUarray tex)
	{
		succeed(cuTexRefSetArray(texf_ref, tex, CU_TRSA_OVERRIDE_FORMAT));
		succeed(cuTexRefSetFlags(texf_ref, CU_TRSF_NORMALIZED_COORDINATES));
		succeed(cuTexRefSetFilterMode(texf_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetMipmapFilterMode(texf_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetAddressMode(texf_ref, 0, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetAddressMode(texf_ref, 1, CU_TR_ADDRESS_MODE_MIRROR));
	}

	void Pipeline::setTexture(CUmipmappedArray tex, float clamp_max)
	{
		succeed(cuTexRefSetMipmappedArray(tex_ref, tex, CU_TRSA_OVERRIDE_FORMAT));
		succeed(cuTexRefSetFlags(tex_ref, CU_TRSF_NORMALIZED_COORDINATES));
		succeed(cuTexRefSetFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetMipmapFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetAddressMode(tex_ref, 0, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetAddressMode(tex_ref, 1, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetMipmapLevelClamp(tex_ref, 0.0f, clamp_max));
	}

	void Pipeline::setTextureSRGB(CUmipmappedArray tex, float clamp_max)
	{
		succeed(cuTexRefSetMipmappedArray(tex_ref, tex, CU_TRSA_OVERRIDE_FORMAT));
		succeed(cuTexRefSetFlags(tex_ref, CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB));
		succeed(cuTexRefSetFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetMipmapFilterMode(tex_ref, CU_TR_FILTER_MODE_LINEAR));
		succeed(cuTexRefSetAddressMode(tex_ref, 0, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetAddressMode(tex_ref, 1, CU_TR_ADDRESS_MODE_WRAP));
		succeed(cuTexRefSetMipmapLevelClamp(tex_ref, 0.0f, clamp_max));
	}

	void Pipeline::setStipple(uint64_t mask)
	{
		succeed(cuMemcpyHtoD(stipple, &mask, sizeof(uint64_t)));
	}


	void Pipeline::bindLitPipelineKernel()
	{
		tris_kernel = &simple_shading_tris_kernel;
		quads_kernel = &simple_shading_quads_kernel;
	}

	void Pipeline::bindTexturedPipelineKernel()
	{
		tris_kernel = &textured_shading_tris_kernel;
		quads_kernel = &textured_shading_quads_kernel;
	}

	void Pipeline::bindVertexHeavyPipelineKernel()
	{
		tris_kernel = &vertex_heavy_tris_kernel;
		quads_kernel = nullptr;
	}

	void Pipeline::bindFragmentHeavyPipelineKernel()
	{
		tris_kernel = &fragment_heavy_tris_kernel;
		quads_kernel = nullptr;
	}

	void Pipeline::bindClipspacePipelineKernel()
	{
		tris_kernel = &clipspace_shading_kernel;
		quads_kernel = nullptr;
	}

	void Pipeline::bindVertexHeavyClipspacePipelineKernel()
	{
		tris_kernel = &vertex_heavy_clipspace_shading_kernel;
		quads_kernel = nullptr;
	}

	void Pipeline::bindFragmentHeavyClipspacePipelineKernel()
	{
		tris_kernel = &fragment_heavy_clipspace_shading_kernel;
		quads_kernel = nullptr;
	}

	void Pipeline::bindEyeCandyPipelineKernel()
	{
		tris_kernel = &eyecandy_shading_kernel;
		quads_kernel = nullptr;
	}

	void Pipeline::bindVertexHeavyEyeCandyPipelineKernel()
	{
		tris_kernel = &vertex_heavy_eyecandy_shading_kernel;
		quads_kernel = nullptr;
	}

	void Pipeline::bindFragmentHeavyEyeCandyPipelineKernel()
	{
		tris_kernel = &fragment_heavy_eyecandy_shading_kernel;
		quads_kernel = nullptr;
	}


	void Pipeline::draw(const PipelineKernel& kernel, CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		geometry_stage.setVertexBuffer(vertices, num_vertices);
		geometry_stage.setIndexBuffer(indices, num_indices);

		succeed(cuEventRecord(begin_draw_event, 0));
		kernel.prepare();
		int num_blocks = kernel.launch();
		succeed(cuEventRecord(end_draw_event, 0));
		instrumentation.record(num_blocks);
		succeed(cuEventSynchronize(end_draw_event));

		float time;
		succeed(cuEventElapsedTime(&time, begin_draw_event, end_draw_event));

		drawing_time += time * 0.001;
	}

	void Pipeline::drawTriangles(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(*tris_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawQuads(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(*quads_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawOcean(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices, bool adaptive, bool wireframe)
	{
		framebuffer.clearColorBufferToTextureF();

		if (adaptive)
			draw(wireframe ? ocean_adaptive_wire_kernel : ocean_adaptive_kernel, vertices, num_vertices, indices, num_indices);
		else
			draw(wireframe ? ocean_static_wire_kernel : ocean_static_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawBlendDemo(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(blend_demo_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawIsoBlendDemo(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(iso_blend_demo_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawIsoStipple(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(iso_stipple_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawGlyphDemo(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		framebuffer.clearColorBufferCheckers(0xFFCCCCCCU, 0xFFFFFFFFU, 4U);
		draw(glyph_demo_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawCheckerboardGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(checkerboard_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawCheckerboardQuadGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(checkerboard_quad_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawCheckerboardFragmentGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(checkerboard_fragment_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::drawCheckerboardQuadFragmentGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices)
	{
		draw(checkerboard_quad_fragment_kernel, vertices, num_vertices, indices, num_indices);
	}

	void Pipeline::upsampleTargetQuad(CUsurfObject target)
	{
		framebuffer.upsampleQuad(target);
	}

	void Pipeline::upsampleTarget(CUsurfObject target)
	{
		framebuffer.upsample(target);
	}

	void Pipeline::reportQueueSizes(PerformanceDataCallback& perf_mon) const
	{
		instrumentation.reportQueueSizes(perf_mon);
	}

	void Pipeline::reportTelemetry(PerformanceDataCallback& perf_mon)
	{
		instrumentation.reportTelemetry(perf_mon);
	}

	double Pipeline::resetDrawingTime()
	{
		double temp = drawing_time;
		drawing_time = 0.0;
		return temp;
	}
}
