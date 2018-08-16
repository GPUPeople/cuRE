


#ifndef INCLUDED_GLRENDERER_RENDERER
#define INCLUDED_GLRENDERER_RENDERER

#pragma once

#include <GL/buffer.h>
#include <GL/framebuffer.h>
#include <GL/event.h>

#include <Renderer.h>

#include "shaders/ColoredShader.h"
#include "shaders/TexturedShader.h"
#include "shaders/LitShader.h"
#include "shaders/TexturedLitShader.h"
#include "shaders/ClipspaceShader.h"
#include "shaders/ClipspaceWireframeShader.h"
#include "shaders/HeavyShaders.h"
#include "shaders/EyeCandyShader.h"
#include "ShaderState.h"


namespace GLRenderer
{
	class PerformanceMonitor
	{
		PerformanceDataCallback* callback;

	public:
		PerformanceMonitor(PerformanceDataCallback* callback);
		~PerformanceMonitor();

		void recordDrawingTime(double t) const;
		void recordMemoryStatus() const;
	};


	class Geometry;

	class Renderer : public ::Renderer, private RendereringContext
	{
	private:
		Renderer(const Renderer&) = delete;
		Renderer& operator =(const Renderer&) = delete;

		Renderer(PerformanceDataCallback* performance_callback);
		~Renderer() = default;

		unsigned int buffer_width;
		unsigned int buffer_height;

		GL::Framebuffer fbo;
		GLuint color_buffer;
		GL::Renderbuffer depth_buffer;

		ColoredShader colored_shader;
		TexturedShader textured_shader;
		LitShader lit_shader;
		TexturedLitShader textured_lit_shader;
		ClipspaceShader clipspace_shader;
		VertexHeavyClipspaceShader vertex_heavy_clipspace_shader;
		FragmentHeavyClipspaceShader fragment_heavy_clipspace_shader;
		ClipspaceWireframeShader clipspace_wireframe_shader;
		HeavyVertexShader heavy_vertex_shader;
		HeavyFragmentShader heavy_fragment_shader;
		EyeCandyShader eyecandy_shader;
		VertexHeavyEyeCandyShader vertex_heavy_eyecandy_shader;
		FragmentHeavyEyeCandyShader fragment_heavy_eyecandy_shader;

		Camera::UniformBuffer camera_params;

		ShaderState shader_state;


		GL::Query drawing_time_query;
		double drawing_time;

		PerformanceMonitor perf_mon;

		void clearColorBuffer(float r, float g, float b, float a) override;
		void clearColorBufferCheckers(std::uint32_t a, std::uint32_t b, unsigned int s) override;
		void clearDepthBuffer(float depth) override;
		void setViewport(float x, float y, float width, float height) override;
		void setUniformf(int index, float v) override;
		void setCamera(const Camera::UniformBuffer& params) override;
		void setObjectTransform(const math::affine_float4x4& M) override;
		void setLight(const math::float3& pos, const math::float3& color) override;
		void finish() override;

		static void* operator new(std::size_t size);
		static void operator delete(void* p);

	public:
		static ::Renderer* __stdcall create(int, PerformanceDataCallback* performance_callback);

		::Geometry* createClipspaceGeometry(const float* position, size_t num_vertices) override;
		::Geometry* createIndexedTriangles(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices) override;
		::Geometry* createIndexedQuads(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices) override;
		::Geometry* createEyeCandyGeometry(const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles) override;
		::Geometry* createCheckerboardGeometry(int type, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles) override;
		::Geometry* create2DTriangles(const float* position, const float* normals, const float* color, size_t num_vertices) override;
		::Geometry* createWaterDemo(const float* position, size_t num_vertices, const uint32_t* indices, size_t num_triangles, float* img_data, uint32_t width, uint32_t height, char* normal_data, uint32_t n_width, uint32_t n_height, uint32_t n_levels) override;
		::Geometry* createIsoBlend(float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) override;
		::Geometry* createGlyphDemo(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) override;
		::Geometry* createIsoStipple(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) override;

		::Texture* createTexture2DRGBA8(size_t width, size_t height, unsigned int levels, const std::uint32_t* data) override;
		::Material* createColoredMaterial(const math::float4& color) override;
		::Material* createLitMaterial(const math::float4& color) override;
		::Material* createTexturedMaterial(GLuint tex, const math::float4& color);
		::Material* createTexturedLitMaterial(GLuint tex, const math::float4& color);
		::Material* createVertexHeavyMaterial(int iterations) override;
		::Material* createFragmentHeavyMaterial(int iterations) override;

		::Material* createClipspaceMaterial() override;
		::Material* createVertexHeavyClipspaceMaterial(int iterations) override;
		::Material* createFragmentHeavyClipspaceMaterial(int iterations) override;

		::Material* createEyeCandyMaterial() override;
		::Material* createVertexHeavyEyeCandyMaterial(int iterations) override;
		::Material* createFragmentHeavyEyeCandyMaterial(int iterations) override;

		void setRenderTarget(GLuint color_buffer, int width, int height) override;
		RendereringContext* beginFrame() override;

		void beginDrawTiming();
		void endDrawTiming();

		void destroy();
	};
}

#endif  // INCLUDED_GLRENDERER_RENDERER
