


#ifndef INCLUDED_CURE_PIPELINE
#define INCLUDED_CURE_PIPELINE

#pragma once

#include <CUDA/event.h>

#include "PipelineKernel.h"
#include "GeometryStage.h"
#include "RasterizationStage.h"
#include "Framebuffer.h"
#include "ShaderState.h"
#include "Instrumentation.h"


namespace cuRE
{
	class PipelineModule;

	class Pipeline
	{
		GeometryStage geometry_stage;
		RasterizationStage rasterization_stage;
		Framebuffer framebuffer;
		ShaderState shader_state;
		CUdeviceptr uniform;

		CUtexref tex_ref;
		CUtexref texf_ref;

		CUdeviceptr stipple;

		PipelineKernel* tris_kernel = nullptr;
		PipelineKernel* quads_kernel = nullptr;

		PipelineKernel simple_shading_tris_kernel;
		PipelineKernel simple_shading_quads_kernel;
		PipelineKernel textured_shading_tris_kernel;
		PipelineKernel textured_shading_quads_kernel;
		PipelineKernel vertex_heavy_tris_kernel;
		PipelineKernel fragment_heavy_tris_kernel;
		PipelineKernel clipspace_shading_kernel;
		PipelineKernel vertex_heavy_clipspace_shading_kernel;
		PipelineKernel fragment_heavy_clipspace_shading_kernel;
		PipelineKernel eyecandy_shading_kernel;
		PipelineKernel vertex_heavy_eyecandy_shading_kernel;
		PipelineKernel fragment_heavy_eyecandy_shading_kernel;
		PipelineKernel water_demo_kernel;
		PipelineKernel blend_demo_kernel;
		PipelineKernel iso_blend_demo_kernel;
		PipelineKernel iso_stipple_kernel;
		PipelineKernel glyph_demo_kernel;

		PipelineKernel checkerboard_quad_fragment_kernel;
		PipelineKernel checkerboard_fragment_kernel;
		PipelineKernel checkerboard_quad_kernel;
		PipelineKernel checkerboard_kernel;

		CU::unique_event begin_draw_event;
		CU::unique_event end_draw_event;

		double drawing_time = 0.0;

		Instrumentation instrumentation;


		void draw(const PipelineKernel& kernel, CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);

	public:
		Pipeline(const Pipeline&) = delete;
		Pipeline& operator =(const Pipeline&) = delete;

		Pipeline(const PipelineModule& module);

		void attachColorBuffer(CUarray color_buffer, int width, int height);
		void attachDepthBuffer(CUarray depth_buffer, int width, int height);

		void clearColorBufferCheckers(uint32_t a, uint32_t b, unsigned int s);
		void clearColorBuffer(float r, float g, float b, float a);
		void clearDepthBuffer(float depth);

		void setViewport(float x, float y, float width, float height);

		void setCamera(const Camera::UniformBuffer& camera);

		void setUniformi(int index, int v);
		void setUniformf(int index, float v);

		void setTexture(CUarray tex);
		void setTextureSRGB(CUarray tex);
		void setTexture(CUmipmappedArray tex, float clamp_max);
		void setTextureSRGB(CUmipmappedArray tex, float clamp_max);
		void setTextureF(CUarray texf);

		void setStipple(uint64_t mask);

		void bindLitPipelineKernel();
		void bindTexturedPipelineKernel();
		void bindVertexHeavyPipelineKernel();
		void bindFragmentHeavyPipelineKernel();
		void bindClipspacePipelineKernel();
		void bindVertexHeavyClipspacePipelineKernel();
		void bindFragmentHeavyClipspacePipelineKernel();
		void bindEyeCandyPipelineKernel();
		void bindVertexHeavyEyeCandyPipelineKernel();
		void bindFragmentHeavyEyeCandyPipelineKernel();

		void drawTriangles(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawQuads(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawWaterDemo(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawBlendDemo(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawIsoBlendDemo(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawIsoStipple(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawGlyphDemo(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawCheckerboardGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawCheckerboardQuadGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawCheckerboardFragmentGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);
		void drawCheckerboardQuadFragmentGeometry(CUdeviceptr vertices, size_t num_vertices, CUdeviceptr indices, size_t num_indices);

		void upsampleTarget(CUsurfObject target);
		void upsampleTargetQuad(CUsurfObject target);

		void reportQueueSizes(PerformanceDataCallback& perf_mon) const;
		void reportTelemetry(PerformanceDataCallback& perf_mon);

		double resetDrawingTime();
	};
}

#endif  // INCLUDED_CURE_PIPELINE
