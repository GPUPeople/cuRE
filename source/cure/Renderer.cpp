


#include <iostream>

#include <CUDA/device.h>
#include <CUDA/context.h>
#include <CUDA/gl_graphics_resource.h>

#include "ResourceImp.h"

#include "Geometry.h"
#include "Texture.h"
#include "materials/LitMaterial.h"
#include "materials/HeavyMaterial.h"
#include "materials/ClipspaceMaterial.h"
#include "materials/EyeCandyMaterial.h"
#include "materials/OceanMaterial.h"

#include "Renderer.h"

#include "utils.h"


namespace
{
	CU::unique_context createContext(CUdevice device)
	{
		//auto cc = cuRE::getCC();

		//succeed(cuInit(0U));
		////CUdevice device = device_ordinal >= 0 ? CU::getDevice(device_ordinal) : CU::findMatchingDevice(std::get<0>(cc), std::get<1>(cc));
		//CUdevice device = CU::getDevice(device_ordinal);
		int num_multiprocessors;
		succeed(cuDeviceGetAttribute(&num_multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

		int threads_per_block;
		succeed(cuDeviceGetAttribute(&threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));

		int regs_per_block;
		succeed(cuDeviceGetAttribute(&regs_per_block, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));

		int smem_per_block;
		succeed(cuDeviceGetAttribute(&smem_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));

		std::cout << "using device " << CU::getDeviceName(device) << "\n  "
		          << num_multiprocessors << " multiprocessors\n  "
		          << threads_per_block << " threads per block\n  "
		          << regs_per_block << " registers per block\n  "
		          << smem_per_block / 1024 << " KB shared memory per block\n\n";

		return CU::createContext(0U, device);
	}

	void reportGPUInfo(PerformanceDataCallback* perf_mon)
	{
		auto context = CU::getCurrentContext();
		auto device = CU::getDevice(context);

		perf_mon->recordGPUInfo(
			CU::getDeviceName(device).c_str(),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_WARP_SIZE>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY>(device),
			CU::getDeviceMemory(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>(device) * 1000,
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK>(device),
			CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>(device));
	}
}

namespace cuRE
{
	PerformanceMonitor::PerformanceMonitor(PerformanceDataCallback* performance_callback)
		: callback(performance_callback)
	{
		recordMemoryStatus();
	}

	PerformanceMonitor::~PerformanceMonitor()
	{
		recordMemoryStatus();
	}

	void PerformanceMonitor::recordMemoryStatus() const
	{
		if (callback)
		{
			size_t total, free;
			succeed(cuMemGetInfo(&free, &total));

			callback->recordMemoryStatus(free, total);
		}
	}

	void PerformanceMonitor::recordDrawingTime(double t) const
	{
		if (callback)
			callback->recordDrawingTime(t);
	}

	void PerformanceMonitor::recordQueueSizes(Pipeline& pipeline) const
	{
		if (callback)
			pipeline.reportQueueSizes(*callback);
	}

	void PerformanceMonitor::reportTelemetry(Pipeline& pipeline) const
	{
		if (callback)
			pipeline.reportTelemetry(*callback);
	}


	Renderer::Renderer(CUdevice device, PerformanceDataCallback* performance_callback)
		: context(createContext(device)),
		  perf_mon(performance_callback),
		  buffer_width(0),
		  buffer_height(0),
		  pipeline(pipeline_module)
	{
		succeed(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 32U*1024*1024));
		reportGPUInfo(performance_callback);
		perf_mon.recordQueueSizes(pipeline);
	}

	::Material* Renderer::createColoredMaterial(const math::float4& color)
	{
		return createLitMaterial(color);
	}

	::Material* Renderer::createLitMaterial(const math::float4& color)
	{
		LitMaterial* mat = ResourceImp<LitMaterial>::create(pipeline, color);
		return mat;
	}

	::Material* Renderer::createVertexHeavyMaterial(int iterations)
	{
		VertexHeavyMaterial* mat = ResourceImp<VertexHeavyMaterial>::create(pipeline, iterations);
		return mat;
	}

	::Material* Renderer::createFragmentHeavyMaterial(int iterations)
	{
		FragmentHeavyMaterial* mat = ResourceImp<FragmentHeavyMaterial>::create(pipeline, iterations);
		return mat;
	}

	::Material* Renderer::createClipspaceMaterial()
	{
		ClipspaceMaterial* mat = ResourceImp<ClipspaceMaterial>::create(pipeline);
		return mat;
	}

	::Material* Renderer::createVertexHeavyClipspaceMaterial(int iterations)
	{
		VertexHeavyClipspaceMaterial* mat = ResourceImp<VertexHeavyClipspaceMaterial>::create(pipeline, iterations);
		return mat;
	}

	::Material* Renderer::createFragmentHeavyClipspaceMaterial(int iterations)
	{
		FragmentHeavyClipspaceMaterial* mat = ResourceImp<FragmentHeavyClipspaceMaterial>::create(pipeline, iterations);
		return mat;
	}

	::Material* Renderer::createEyeCandyMaterial()
	{
		EyeCandyMaterial* mat = ResourceImp<EyeCandyMaterial>::create(pipeline);
		return mat;
	}

	::Material* Renderer::createVertexHeavyEyeCandyMaterial(int iterations)
	{
		VertexHeavyEyeCandyMaterial* mat = ResourceImp<VertexHeavyEyeCandyMaterial>::create(pipeline, iterations);
		return mat;
	}

	::Material* Renderer::createFragmentHeavyEyeCandyMaterial(int iterations)
	{
		FragmentHeavyEyeCandyMaterial* mat = ResourceImp<FragmentHeavyEyeCandyMaterial>::create(pipeline, iterations);
		return mat;
	}

	::Material* Renderer::createOceanMaterial(const void* img_data, size_t width, size_t height, const void* normal_data, size_t n_width, size_t n_height, unsigned int n_levels)
	{
		OceanMaterial* mat = ResourceImp<OceanMaterial>::create(pipeline, static_cast<const float*>(img_data), width, height, static_cast<const std::uint32_t*>(normal_data), n_width, n_height, n_levels);
		return mat;
	}

	::Geometry* Renderer::createIndexedTriangles(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	{
		auto geom = ResourceImp<IndexedTriangles>::create(pipeline, position, normals, texcoord, num_vertices, indices, num_indices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createIndexedQuads(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	{
		auto geom = ResourceImp<IndexedQuads>::create(pipeline, position, normals, texcoord, num_vertices, indices, num_indices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createOceanGeometry(const float* position, size_t num_vertices, const uint32_t* indices, size_t num_indices)
	{
		auto geom = ResourceImp<OceanGeometry>::create(pipeline, position, num_vertices, indices, num_indices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createClipspaceGeometry(const float* position, size_t num_vertices)
	{
		auto geom = ResourceImp<ClipspaceGeometry>::create(pipeline, position, num_vertices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::create2DTriangles(const float* position, const float* normals, const float* color, size_t num_vertices)
	{
		auto geom = ResourceImp<BlendGeometry>::create(pipeline, position, normals, color, num_vertices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createIsoBlend(float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices)
	{
		auto geom = ResourceImp<IsoBlendGeometry>::create(pipeline, vert_data, num_vertices, index_data, num_indices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createGlyphDemo(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices)
	{
		auto geom = ResourceImp<GlyphGeometry>::create(pipeline, mask, vert_data, num_vertices, index_data, num_indices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createIsoStipple(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices)
	{
		auto geom = ResourceImp<IsoStippleGeometry>::create(pipeline, mask, vert_data, num_vertices, index_data, num_indices);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Texture* Renderer::createTexture2DRGBA8(size_t width, size_t height, unsigned int levels, const std::uint32_t* data)
	{
		auto tex = ResourceImp<Texture>::create(pipeline, width, height, levels, data);
		perf_mon.recordMemoryStatus();
		return tex;
	}

	::Geometry* Renderer::createEyeCandyGeometry(const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	{
		auto geom = ResourceImp<EyeCandyGeometry>::create(pipeline, position, num_vertices, indices, triangle_colors, num_triangles);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createCheckerboardGeometry(int type, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	{
		auto geom = ResourceImp<CheckerboardGeometry>::create(pipeline, type, position, num_vertices, indices, triangle_colors, num_triangles);
		need_target_upsampling = (type & 0x4U) != 0U;
		perf_mon.recordMemoryStatus();
		return geom;
	}

	void Renderer::setRenderTarget(GLuint color_buffer, int width, int height)
	{
		color_buffer_resource = CU::graphics::registerGLImage(color_buffer, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD | CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);

		if (buffer_width != width || buffer_height != height)
		{
			buffer_width = width;
			buffer_height = height;

			if (need_target_upsampling)
				target_color_buffer = CU::createArray2D(width, height, CU_AD_FORMAT_UNSIGNED_INT8, 4, CUDA_ARRAY3D_SURFACE_LDST);
			
			//TODO: make sure this size is coherent with the rasterization space!
			unsigned int dp_width = divup(width, 128) * 128;
			unsigned int dp_height = divup(height, 128) * 128;

			CUDA_ARRAY3D_DESCRIPTOR depthdesc = { dp_width, dp_height, 1, CU_AD_FORMAT_FLOAT, 1, CUDA_ARRAY3D_SURFACE_LDST};
			depth_buffer = CU::createArray(depthdesc);
			//depth_buffer = CU::allocMemory(dp_width*dp_height * 4U);

			pipeline.attachDepthBuffer(depth_buffer, dp_width, dp_height);
		}
	}


	void Renderer::setViewport(float x, float y, float width, float height)
	{
		pipeline.setViewport(x, y, width, height);
	}


	RendereringContext* Renderer::beginFrame()
	{
		CUgraphicsResource resources[] = { color_buffer_resource };
		succeed(cuGraphicsMapResources(1U, resources, 0));
		mapped_color_buffer = CU::graphics::getMappedArray(color_buffer_resource, 0U, 0U);

		if (target_color_buffer)
		{
			upsample_target.reset();
			upsample_target = CU::createSurfaceObject(mapped_color_buffer);
			pipeline.attachColorBuffer(target_color_buffer, buffer_width, buffer_height);
		}
		else
			pipeline.attachColorBuffer(mapped_color_buffer, buffer_width, buffer_height);

		perf_mon.recordMemoryStatus();

		return this;
	}

	void Renderer::finish()
	{
		if (need_target_upsampling)
			pipeline.upsampleTargetQuad(upsample_target);

		CUgraphicsResource resources[] = { color_buffer_resource };
		succeed(cuGraphicsUnmapResources(1U, resources, 0));

		perf_mon.recordDrawingTime(pipeline.resetDrawingTime());
		perf_mon.reportTelemetry(pipeline);
	}


	void Renderer::clearColorBuffer(float r, float g, float b, float a)
	{
		pipeline.clearColorBuffer(r, g, b, a);
	}

	void Renderer::clearColorBufferCheckers(std::uint32_t a, std::uint32_t b, unsigned int s)
	{
		pipeline.clearColorBufferCheckers(a, b, s);
	}

	void Renderer::clearDepthBuffer(float depth)
	{
		pipeline.clearDepthBuffer(depth);
	}


	void Renderer::setCamera(const Camera::UniformBuffer& params)
	{
		camera_params = params;
		camera_params.PVM = camera_params.PV * model_matrix;
		camera_params.PVM_inv = inverse(camera_params.PVM);
		pipeline.setCamera(camera_params);
	}

	void Renderer::setObjectTransform(const math::affine_float4x4& M)
	{
		model_matrix = M;
		camera_params.PVM = camera_params.PV * model_matrix;
		camera_params.PVM_inv = inverse(camera_params.PVM);
		pipeline.setCamera(camera_params);
	}

	void Renderer::setLight(const math::float3& pos, const math::float3& color)
	{
		light_pos = pos;
		light_color = color;
	}

	void Renderer::setUniformf(int index, float v)
	{
		pipeline.setUniformf(index, v);
	}

	::Renderer* __stdcall Renderer::create(CUdevice device, PerformanceDataCallback* performance_callback)
	{
		return new Renderer(device, performance_callback);
	}

	void Renderer::destroy()
	{
		delete this;
	}

	void* Renderer::operator new(std::size_t size)
	{
		auto p = _aligned_malloc(size, __alignof(Renderer));
		if (p == nullptr)
			throw std::bad_alloc();
		return p;
	}

	void Renderer::operator delete(void* p)
	{
		_aligned_free(p);
	}
}
