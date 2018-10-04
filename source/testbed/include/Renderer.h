


#ifndef INCLUDED_RENDERER
#define INCLUDED_RENDERER

#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>

#include <GL/gl.h>

#include <math/matrix.h>

#include "Camera.h"
#include "PlugIn.h"
#include "Resource.h"



struct TimingResult
{
	std::uint64_t t;
	std::uint64_t N;
};

struct INTERFACE PerformanceDataCallback
{
	virtual void recordGPUInfo(const char* name, int compute_capability_major, int compute_capability_minor, int num_multiprocessors, int warp_size, int max_threads_per_mp, int regs_per_mp, std::size_t shared_memory_per_mp, std::size_t total_constant_memory, std::size_t total_global_memory, int clock_rate, int max_threads_per_block, int max_regs_per_block, std::size_t max_shared_memory_per_block) = 0;
	virtual void recordDrawingTime(double t) = 0;
	virtual void recordInstrumentationTimers(std::unique_ptr<TimingResult[]> timers, int num_timers, int num_multiprocessors) = 0;
	virtual void recordMemoryStatus(std::size_t free, std::size_t total) = 0;
	virtual void recordQueueSize(const std::uint32_t* queue_size, int num_queues) = 0;
	virtual void recordMaxQueueFillLevel(const std::uint32_t* max_fill_level, int num_queues) = 0;

protected:
	PerformanceDataCallback() = default;
	PerformanceDataCallback(const PerformanceDataCallback&) = default;
	PerformanceDataCallback& operator=(const PerformanceDataCallback&) = default;
	~PerformanceDataCallback() = default;
};

struct INTERFACE RendereringContext
{
	virtual void clearColorBuffer(float r, float g, float b, float a) = 0;
	virtual void clearColorBufferCheckers(std::uint32_t a, std::uint32_t b, unsigned int s) = 0;
	virtual void clearDepthBuffer(float depth) = 0;
	virtual void setViewport(float x, float y, float width, float height) = 0;
	virtual void setUniformf(int index, float v) = 0;
	virtual void setCamera(const Camera::UniformBuffer& params) = 0;
	virtual void setObjectTransform(const math::affine_float4x4& M) = 0;
	virtual void setLight(const math::float3& pos, const math::float3& color) = 0;
	virtual void finish() = 0;

protected:
	RendereringContext() = default;
	RendereringContext(const RendereringContext&) = default;
	RendereringContext& operator=(const RendereringContext&) = default;
	~RendereringContext() = default;
};

struct INTERFACE Renderer : public virtual PlugIn
{
	virtual Geometry* createClipspaceGeometry(const float* position, size_t num_vertices) = 0;
	virtual Geometry* createIndexedTriangles(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices) = 0;
	virtual Geometry* createIndexedQuads(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices) = 0;
	virtual Geometry* createEyeCandyGeometry(const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles) = 0;
	virtual Geometry* createOceanGeometry(const float* position, size_t num_vertices, const uint32_t* indices, size_t num_triangles) = 0;
	virtual Geometry* createCheckerboardGeometry(int type, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles) = 0;
	virtual Geometry* create2DTriangles(const float* position, const float* normals, const float* color, size_t num_vertices) = 0;
	virtual Geometry* createIsoBlend(float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) = 0;
	virtual Geometry* createGlyphDemo(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) = 0;
	virtual Geometry* createIsoStipple(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices) = 0;

	virtual Texture* createTexture2DRGBA8(size_t width, size_t height, unsigned int levels, const std::uint32_t* data) = 0;
	virtual Material* createColoredMaterial(const math::float4& color) = 0;
	virtual Material* createLitMaterial(const math::float4& color) = 0;
	virtual Material* createVertexHeavyMaterial(int iterations) = 0;
	virtual Material* createFragmentHeavyMaterial(int iterations) = 0;

	virtual Material* createClipspaceMaterial() = 0;
	virtual Material* createVertexHeavyClipspaceMaterial(int iterations) = 0;
	virtual Material* createFragmentHeavyClipspaceMaterial(int iterations) = 0;

	virtual Material* createEyeCandyMaterial() = 0;
	virtual Material* createVertexHeavyEyeCandyMaterial(int iterations) = 0;
	virtual Material* createFragmentHeavyEyeCandyMaterial(int iterations) = 0;

	virtual Material* createOceanMaterial(const void* img_data, size_t width, size_t height, const void* normal_data, size_t n_width, size_t n_height, unsigned int n_levels) = 0;

	virtual void setRenderTarget(GLuint color_buffer, int width, int height) = 0;
	virtual RendereringContext* beginFrame() = 0;

protected:
	Renderer() = default;
	Renderer(const Renderer&) = default;
	Renderer& operator=(const Renderer&) = default;
	~Renderer() = default;
};

typedef Renderer*(__stdcall* createRendererFunc)(int, PerformanceDataCallback*);

#endif // INCLUDED_RENDERER
