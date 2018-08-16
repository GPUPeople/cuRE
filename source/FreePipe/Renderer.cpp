


#include <stdexcept>
#include <iostream>

//#include <CUDA/binary.h>
#include <CUDA/device.h>
#include <CUDA/gl_graphics_resource.h>

#include "ResourceImp.h"

#include "Geometry.h"
#include "Texture.h"
#include "materials/LitMaterial.h"
#include "materials/ColoredMaterial.h"
#include "materials/ClipspaceMaterial.h"
#include "Renderer.h"


namespace CUBIN
{
	extern const char free_pipe;
	extern const char free_pipe_end;
}

namespace
{
	CU::unique_context createContext(CUdevice device)
	{
		//auto cc = CU::readComputeCapability(&CUBIN::free_pipe);

		//succeed(cuInit(0U));
		//CUdevice device = device_ordinal >= 0 ? CU::getDevice(device_ordinal) : CU::findMatchingDevice(std::get<0>(cc), std::get<1>(cc));
		std::cout << "using device " << CU::getDeviceName(device) << "\n";
		return CU::createContext(0U, device);
	}

	CU::unique_module createModule()
	{
		return CU::loadModule(&CUBIN::free_pipe);
	}

	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace FreePipe
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


	Renderer::Renderer(CUdevice device, PerformanceDataCallback* performance_callback)
		: context(createContext(device)),
		  perf_mon(performance_callback),
		  module(createModule()),
		  buffer_width(0), buffer_height(0),
		  begin_geometry_event(CU::createEvent()),
		  end_geometry_event(CU::createEvent()),
		  begin_rasterization_event(CU::createEvent()),
		  end_rasterization_event(CU::createEvent()),
		  vertex_shader(*this, module)
	{
		// get constants
		size_t constant_size;

		succeed(cuModuleGetSurfRef(&color_buffer_surface_reference, module, "color_buffer"));
		succeed(cuModuleGetGlobal(&pc_bufferDims, &constant_size, module, "c_bufferDims"));


		succeed(cuModuleGetGlobal(&pc_depthBuffer, &constant_size, module, "c_depthBuffer"));
		succeed(cuModuleGetGlobal(&viewport, &constant_size, module, "viewport"));
		succeed(cuModuleGetGlobal(&pixel_step, &constant_size, module, "pixel_step"));

		// get kernels
		succeed(cuModuleGetFunction(&kernel_clear_color, module, "clearColorBuffer"));
		succeed(cuModuleGetFunction(&kernel_clear_depth, module, "resetDepthStorage"));


		succeed(cuModuleGetGlobal(&pc_vertex_matrix, &constant_size, module, "c_VertexTransformMatrix"));
		succeed(cuModuleGetGlobal(&pc_model_matrix, &constant_size, module, "c_ModelMatrix"));
		succeed(cuModuleGetGlobal(&pc_normal_matrix, &constant_size, module, "c_NormalTransformMatrix"));
	}

	::Geometry* Renderer::createIndexedTriangles(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const unsigned int* indices, size_t num_indices)
	{
		IndexedGeometry* geom = ResourceImp<IndexedGeometry>::create(*this, module, position, normals, texcoord, num_vertices, indices, num_indices);
		return geom;
	}

	::Geometry* Renderer::createIndexedQuads(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const unsigned int* indices, size_t num_indices)
	{
		throw std::runtime_error("primitive type not supported");
	}

	::Geometry* Renderer::createClipspaceGeometry(const float* position, size_t num_vertices)
	{
		ClipspaceGeometry* geom = ResourceImp<ClipspaceGeometry>::create(*this, module, position, num_vertices);
		return geom;
	}

	::Texture* Renderer::createTexture2DRGBA8(size_t width, size_t height, unsigned int levels, const std::uint32_t* data)
	{
		return ResourceImp<Texture>::create(*this, width, height, levels, data);
	}

	::Material* Renderer::createColoredMaterial(const math::float4& color)
	{
		ColoredMaterial* mat = ResourceImp<ColoredMaterial>::create(*this, color, module);
		return mat;
	}

	::Material* Renderer::createLitMaterial(const math::float4& color)
	{
		LitMaterial* mat = ResourceImp<LitMaterial>::create(*this, color, module);
		return mat;
	}

	::Material* Renderer::createVertexHeavyMaterial(int iterations)
	{
		throw std::runtime_error("no heavy materials here!");
	}

	::Material* Renderer::createFragmentHeavyMaterial(int iterations)
	{
		throw std::runtime_error("no heavy materials here!");
	}

	::Material* Renderer::createClipspaceMaterial()
	{
		ClipspaceMaterial* mat = ResourceImp<ClipspaceMaterial>::create(*this, module);
		return mat;
	}

	::Material* Renderer::createVertexHeavyClipspaceMaterial(int iterations)
	{
		throw std::runtime_error("no heavy materials here!");
	}

	::Material* Renderer::createFragmentHeavyClipspaceMaterial(int iterations)
	{
		throw std::runtime_error("no heavy materials here!");
	}

	::Geometry* Renderer::createEyeCandyGeometry(const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	{
		throw std::runtime_error("Help! Eye candy not implemented for this renderer!");
	}

	::Material* Renderer::createEyeCandyMaterial()
	{
		throw std::runtime_error("Help! Eye candy not implemented for this renderer!");
	}

	::Material* Renderer::createVertexHeavyEyeCandyMaterial(int iterations)
	{
		throw std::runtime_error("no heavy materials here!");
	}

	::Material* Renderer::createFragmentHeavyEyeCandyMaterial(int iterations)
	{
		throw std::runtime_error("no heavy materials here!");
	}

	::Geometry* Renderer::createCheckerboardGeometry(int type, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	{
		throw std::runtime_error("Help! Checker board not implemented for this renderer!");
	}

	::Geometry* Renderer::create2DTriangles(const float* position, const float* normals, const float* color, size_t num_vertices)
	{
		throw std::runtime_error("Help! 2D triangles not implemented for this renderer!");
	}

	::Geometry* Renderer::createWaterDemo(const float* position, size_t num_vertices, const uint32_t* indices, size_t num_triangles, float* img_data, uint32_t width, uint32_t height, char* normal_data, uint32_t n_width, uint32_t n_height, uint32_t n_levels)
	{
		throw std::runtime_error("Help! Water demo not implemented for this renderer!");
	}

	::Geometry* Renderer::createIsoBlend(float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices)
	{
		throw std::runtime_error("Help! Isoblend demo not implemented for this renderer!");
	}

	::Geometry* Renderer::createGlyphDemo(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices)
	{
		throw std::runtime_error("Help! Glyph demo not implemented for this renderer!");
	}

	::Geometry* Renderer::createIsoStipple(uint64_t mask, float* vert_data, uint32_t num_vertices, uint32_t* index_data, uint32_t num_indices)
	{
		throw std::runtime_error("Help! Isostipple not implemented for this renderer!");
	}


	void Renderer::setRenderTarget(GLuint color_buffer, int width, int height)
	{
		color_buffer_resource = CU::graphics::registerGLImage(color_buffer, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD | CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
		
		if (buffer_width != width || buffer_height != height)
		{ 
			buffer_width = static_cast<unsigned int>(width);
			buffer_height = static_cast<unsigned int>(height);
			unsigned int size[] = { buffer_width, buffer_height };
			succeed(cuMemcpyHtoD(pc_bufferDims, size, sizeof(size)));

			depthBuffer = CU::allocMemory(width*height * 4u);
			CUdeviceptr temp = depthBuffer;
			succeed(cuMemcpyHtoD(pc_depthBuffer, &temp, sizeof(CUdeviceptr)));

			setViewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
		}
		
	}


	void Renderer::setViewport(float x, float y, float width, float height)
	{
		float vp[] = { x, y, x + width, y + height };
		succeed(cuMemcpyHtoD(viewport, vp, sizeof(vp)));

		float inv_pixstep[] = { 2.0f / width, 2.0f / height };
		succeed(cuMemcpyHtoD(pixel_step, inv_pixstep, sizeof(inv_pixstep)));
	}

	RendereringContext* Renderer::beginFrame()
	{
		CUgraphicsResource resources[] = { color_buffer_resource };
		succeed(cuGraphicsMapResources(1U, resources, 0));
		mapped_color_buffer = CU::graphics::getMappedArray(color_buffer_resource, 0U, 0U);
		succeed(cuSurfRefSetArray(color_buffer_surface_reference, mapped_color_buffer, 0U));

		geometry_time = 0.0;
		rasterization_time = 0.0;

		perf_mon.recordMemoryStatus();

		return this;
	}

	void Renderer::finish()
	{
		CUgraphicsResource resources[] = { color_buffer_resource };
		succeed(cuGraphicsUnmapResources(1U, resources, 0));

		perf_mon.recordDrawingTime(geometry_time + rasterization_time);
	}


	void Renderer::clearColorBuffer(float r, float g, float b, float a)
	{
		// clear color buffer
		const unsigned int block_size_x = 16;
		const unsigned int block_size_y = 16;
		unsigned int num_blocks_x = divup(buffer_width, block_size_x);
		unsigned int num_blocks_y = divup(buffer_height, block_size_y);

		unsigned char color[4] = { static_cast<unsigned char>(r * 255), static_cast<unsigned char>(g * 255), static_cast<unsigned char>(b * 255), static_cast<unsigned char>(a * 255) };
		void* params[] = {
			color,
			&buffer_width,
			&buffer_height
		};

		succeed(cuLaunchKernel(kernel_clear_color, num_blocks_x, num_blocks_y, 1, block_size_x, block_size_y, 1, 0, 0, params, nullptr));
		succeed(cuCtxSynchronize());
	}

	void Renderer::clearColorBufferCheckers(std::uint32_t a, std::uint32_t b, unsigned int s)
	{
		throw std::runtime_error("not implemented here");
	}

	void Renderer::clearDepthBuffer(float depth)
	{
		succeed(cuMemsetD32(depthBuffer, *reinterpret_cast<unsigned int*>(&depth), buffer_width * buffer_height));
	}


	void Renderer::setUniformf(int index, float v)
	{
	}

	void Renderer::updatePVM()
	{
		math::float4x4 PVM = camera_params.PV * model_matrix;
		math::float4x4 N = model_matrix;
		math::float4x4 M = model_matrix;

		succeed(cuMemcpyHtoD(pc_vertex_matrix, &PVM._11, 4 * 16));
		succeed(cuMemcpyHtoD(pc_normal_matrix, &N._11, 4 * 16));
		succeed(cuMemcpyHtoD(pc_model_matrix, &M, 4 * 16));
	}

	void Renderer::setCamera(const Camera::UniformBuffer& params)
	{
		camera_params = params;
		updatePVM();
	}

	void Renderer::setObjectTransform(const math::affine_float4x4& M)
	{
		model_matrix = M;
		updatePVM();
	}

	void Renderer::setLight(const math::float3& pos, const math::float3& color)
	{
		light_pos = pos;
		light_color = color;
	}

	void Renderer::runVertexShader(unsigned int num_patches, unsigned int num_vertices, unsigned int num_indices, bool light, bool texcoords)
	{
		vertex_shader.run(num_patches, num_vertices, num_indices, light, texcoords);
	}


	void Renderer::beginTimingGeometry()
	{
		succeed(cuEventRecord(begin_geometry_event, 0));
	}

	void Renderer::endTimingGeometry()
	{
		succeed(cuEventRecord(end_geometry_event, 0));
		cuEventSynchronize(end_geometry_event);

		float t;
		cuEventElapsedTime(&t, begin_geometry_event, end_geometry_event);

		geometry_time += t * 0.001;
	}

	void Renderer::beginTimingRasterization()
	{
		succeed(cuEventRecord(begin_rasterization_event, 0));
	}

	void Renderer::endTimingRasterization()
	{
		succeed(cuEventRecord(end_rasterization_event, 0));
		cuEventSynchronize(end_rasterization_event);

		float t;
		cuEventElapsedTime(&t, begin_rasterization_event, end_rasterization_event);

		rasterization_time += t * 0.001;
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
