


#ifndef INCLUDED_FREEPIPE_RENDERER
#define INCLUDED_FREEPIPE_RENDERER

#pragma once

#include <memory>

#include <CUDA/context.h>
#include <CUDA/module.h>
#include <CUDA/memory.h>
#include <CUDA/event.h>
#include <CUDA/graphics_resource.h>

#include <Renderer.h>

#include "VertexShader.h"


namespace FreePipe
{
	class PerformanceMonitor
	{
		PerformanceDataCallback* callback;

	public:
		PerformanceMonitor(PerformanceDataCallback* callback);
		~PerformanceMonitor();

		void recordMemoryStatus() const;
		void recordDrawingTime(double t) const;
	};


	class Renderer : public ::Renderer, private RendereringContext
	{
	private:
		Renderer(const Renderer&) = delete;
		Renderer& operator =(const Renderer&) = delete;

		Renderer(CUdevice device, PerformanceDataCallback* performance_callback);
		~Renderer() = default;

		CU::unique_context context;

		PerformanceMonitor perf_mon;

		CU::unique_module module;

		CU::graphics::unique_resource color_buffer_resource;
		CUarray mapped_color_buffer;

		CU::unique_ptr depthBuffer;

		Camera::UniformBuffer camera_params;

		math::affine_float4x4 model_matrix;

		math::float3 light_pos;
		math::float3 light_color;


		CUdeviceptr pc_depthBuffer;
		CUdeviceptr pc_bufferDims;
		CUdeviceptr pixel_step;

		CUdeviceptr viewport;
		CUsurfref color_buffer_surface_reference;

		CUfunction kernel_clear_color;
		CUfunction kernel_clear_depth;

		unsigned int buffer_width;
		unsigned int buffer_height;


		CUdeviceptr pc_vertex_matrix;
		CUdeviceptr pc_model_matrix;
		CUdeviceptr pc_normal_matrix;

		VertexShader vertex_shader;


		CU::unique_event begin_geometry_event;
		CU::unique_event end_geometry_event;
		CU::unique_event begin_rasterization_event;
		CU::unique_event end_rasterization_event;
		double geometry_time;
		double rasterization_time;

		void clearColorBuffer(float r, float g, float b, float a) override;
		void clearColorBufferCheckers(std::uint32_t a, std::uint32_t b, unsigned int s) override;
		void clearDepthBuffer(float depth) override;
		void setViewport(float x, float y, float width, float height) override;
		void setUniformf(int index, float v) override;
		void setCamera(const Camera::UniformBuffer& params) override;
		void setObjectTransform(const math::affine_float4x4& M) override;
		void setLight(const math::float3& pos, const math::float3& color) override;
		void finish() override;

		void updatePVM();

	public:
		static ::Renderer* __stdcall create(CUdevice device, PerformanceDataCallback* performance_callback);

		::Geometry* createClipspaceGeometry(const float* position, size_t num_vertices) override;
		::Geometry* createIndexedTriangles(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices) override;
		::Geometry* createIndexedQuads(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices) override;
		::Geometry* createEyeCandyGeometry(const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles) override { return nullptr; }
		::Geometry* createOceanGeometry(const float* position, size_t num_vertices, const uint32_t* indices, size_t num_triangles) { return nullptr; }
		::Geometry* createCheckerboardGeometry(int type, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles) override { return nullptr; }
		::Geometry* create2DTriangles(const float* position, const float* normals, const float* color, size_t num_vertices) override { return nullptr; }
		::Geometry* createIsoBlend(float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) override { return nullptr; }
		::Geometry* createGlyphDemo(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) override { return nullptr; }
		::Geometry* createIsoStipple(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) override { return nullptr; }

		::Texture* createTexture2DRGBA8(size_t width, size_t height, unsigned int levels, const std::uint32_t* data) override;
		::Material* createColoredMaterial(const math::float4& color) override;
		::Material* createLitMaterial(const math::float4& color) override;
		::Material* createVertexHeavyMaterial(int iterations) override { return nullptr; }
		::Material* createFragmentHeavyMaterial(int iterations) override { return nullptr; }

		::Material* createClipspaceMaterial() override;
		::Material* createVertexHeavyClipspaceMaterial(int iterations) override { return nullptr; }
		::Material* createFragmentHeavyClipspaceMaterial(int iterations) override { return nullptr; }

		::Material* createEyeCandyMaterial() override { return nullptr; }
		::Material* createVertexHeavyEyeCandyMaterial(int iterations) override { return nullptr; }
		::Material* createFragmentHeavyEyeCandyMaterial(int iterations) override { return nullptr; }

		::Material* createOceanMaterial(const void* img_data, size_t width, size_t height, const void* normal_data, size_t n_width, size_t n_height, unsigned int n_levels) override { return nullptr; }

		void setRenderTarget(GLuint color_buffer, int width, int height) override;
		RendereringContext* beginFrame() override;


		void runVertexShader(unsigned int num_patches, unsigned int num_vertices, unsigned int num_indices, bool light = true, bool texcoords = true);


		const math::float3& getLightPos() { return light_pos; }
		const math::float3& getLightColor() { return light_color; }

		void beginTimingGeometry();
		void endTimingGeometry();
		void beginTimingRasterization();
		void endTimingRasterization();

		CUmodule getModule() {	return module;	}

		void destroy();
	};
}

#endif  // INCLUDED_FREEPIPE_RENDERER
