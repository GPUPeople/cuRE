


#include <stdexcept>

//#include <CUDA/binary.h>
#include <CUDA/device.h>
#include <CUDA/gl_graphics_resource.h>

#include "ResourceImp.h"

#include "Geometry.h"
#include "Renderer.h"
#include "materials/LitMaterial.h"
#include "materials/ColoredMaterial.h"
#include "materials/ClipspaceMaterial.h"
#include "materials/EyeCandyMaterial.h"
//#include "Texture.h"
//#include "shaders/vertex_simple.h"
//#include "shaders/fragment_phong.h"


namespace CUBIN
{
	extern const char cudaraster;
}

namespace
{
	CU::unique_context createContext(CUdevice device)
	{
		//auto cc = CU::readComputeCapability(&CUBIN::cudaraster);

		//succeed(cuInit(0U));
		//CUdevice device = device_ordinal >= 0 ? CU::getDevice(device_ordinal) : CU::findMatchingDevice(std::get<0>(cc), std::get<1>(cc));
		std::cout << "using device " << CU::getDeviceName(device) << "\n";
		return CU::createContext(0U, device);
	}

	CU::unique_module createModule()
	{
		return CU::loadModule(&CUBIN::cudaraster);
	}

	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace CUDARaster
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
		  app(module),
		  clear_color(FW::Vec4f(0,0,0,0)),
		  clear_depth(1)
	{
	}

	::Geometry* Renderer::createIndexedTriangles(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	{
		auto geom = ResourceImp<IndexedGeometry>::create(this, position, normals, texcoord, num_vertices, indices, num_indices, app);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createIndexedQuads(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	{
		throw std::runtime_error("primitive type not supported");
	}

	::Geometry* Renderer::createClipspaceGeometry(const float* position, size_t num_vertices)
	{
		auto geom = ResourceImp<ClipspaceGeometry>::create(this, position, num_vertices, app);
		perf_mon.recordMemoryStatus();
		return geom;
	}

	::Geometry* Renderer::createEyeCandyGeometry(const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	{
		auto geom = ResourceImp<EyeCandyGeometry>::create(this, position, num_vertices, indices, triangle_colors, num_triangles, app);
		perf_mon.recordMemoryStatus();
		return geom;
	}


	::Texture* Renderer::createTexture2DRGBA8(size_t width, size_t height, unsigned int levels, const std::uint32_t* data)
	{
		return nullptr; 
	}

	::Material* Renderer::createLitMaterial(const math::float4& color)
	{
		auto mat = ResourceImp<LitMaterial>::create(color, module);
		perf_mon.recordMemoryStatus();
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

	::Material* Renderer::createColoredMaterial(const math::float4& color)
	{
		auto mat = ResourceImp<ColoredMaterial>::create(color, module);
		perf_mon.recordMemoryStatus();
		return mat;
	}

	::Material* Renderer::createClipspaceMaterial()
	{
		auto mat = ResourceImp<ClipspaceMaterial>::create(module);
		perf_mon.recordMemoryStatus();
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

	::Material* Renderer::createEyeCandyMaterial()
	{
		auto mat = ResourceImp<EyeCandyMaterial>::create(module);
		perf_mon.recordMemoryStatus();
		return mat;
	}


	void Renderer::setRenderTarget(GLuint color_buffer, int width, int height)
	{
		color_buffer_resource = CU::graphics::registerGLImage(color_buffer, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD | CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);

		if (buffer_width != width || buffer_height != height)
		{
			if(buffer_width != 0)
			{	cuArrayDestroy(depth_buffer);	}

			buffer_width = width;
			buffer_height = height;

			CUDA_ARRAY3D_DESCRIPTOR depthdesc;
			depthdesc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
			depthdesc.Depth = 0;
			depthdesc.Width = width;
			depthdesc.Height = height;
			depthdesc.NumChannels = 1;
			depthdesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
			succeed(cuArray3DCreate(&depth_buffer, &depthdesc));
		}
	}

	RendereringContext* Renderer::beginFrame()
	{
		CUgraphicsResource resources[] = { color_buffer_resource };
		succeed(cuGraphicsMapResources(1U, resources, 0));
		mapped_color_buffer = CU::graphics::getMappedArray(color_buffer_resource, 0U, 0U);
		app.setTargets(mapped_color_buffer, depth_buffer, buffer_width, buffer_height);

		rendering_time = 0.0;

		perf_mon.recordMemoryStatus();

		return this;
	}

	void Renderer::recordDrawingTime(double t)
	{
		rendering_time += t;
	}

	void Renderer::finish() 
	{
		app.setClear(false);
		succeed(cuCtxSynchronize());
		CUgraphicsResource resources[] = { color_buffer_resource };
		succeed(cuGraphicsUnmapResources(1U, resources, 0));

		perf_mon.recordDrawingTime(rendering_time);
	}






	void Renderer::clearColorBuffer(float r, float g, float b, float a)
	{
		clear_color = FW::Vec4f(r,g,b,a);
		//app.setClear(true, clear_color, clear_depth);
		app.immediateClearColor(clear_color);
	}


	void Renderer::clearColorBufferCheckers(std::uint32_t a, std::uint32_t b, unsigned int s)
	{
		throw std::runtime_error("not implemented here");
	}

	void Renderer::clearDepthBuffer(float depth)
	{
		clear_depth = depth;
		//app.setClear(true, clear_color, clear_depth);
		app.immediateClearDepth(clear_depth);
	}

	void Renderer::setViewport(float x, float y, float width, float height)
	{
		//TODO
	}

	void Renderer::setUniformf(int index, float v)
	{
	}

	void convert(const math::affine_float4x4& M, FW::Mat4f& dest)
	{
		dest.m00 = M._11;
		dest.m01 = M._12;
		dest.m02 = M._13;
		dest.m03 = M._14;
		dest.m10 = M._21;
		dest.m11 = M._22;
		dest.m12 = M._23;
		dest.m13 = M._24;
		dest.m20 = M._31;
		dest.m21 = M._32;
		dest.m22 = M._33;
		dest.m23 = M._34;
		dest.m30 = 0;
		dest.m31 = 0;
		dest.m32 = 0;
		dest.m33 = 1;
	}

	void convert(const math::float4x4& M, FW::Mat4f& dest)
	{
		dest.m00 = M._11;
		dest.m01 = M._12;
		dest.m02 = M._13;
		dest.m03 = M._14;
		dest.m10 = M._21;
		dest.m11 = M._22;
		dest.m12 = M._23;
		dest.m13 = M._24;
		dest.m20 = M._31;
		dest.m21 = M._32;
		dest.m22 = M._33;
		dest.m23 = M._34;
		dest.m30 = M._41;
		dest.m31 = M._42;
		dest.m32 = M._43;
		dest.m33 = M._44;
	}

	void Renderer::setCamera(const Camera::UniformBuffer& params)
	{
		convert(params.V, view);
		FW::Mat4f comb = view * model;
		app.setPosToCam(comb);

		FW::Mat4f proj;
		convert(params.P, proj);
		app.setProjection(proj);
	}


	void Renderer::setObjectTransform(const math::affine_float4x4& M)
	{
		convert(M, model);
		FW::Mat4f comb = view * model;
		app.setPosToCam(comb);
		app.setModel(model);

		FW::Vec3f fake_light = (model.inverted() * FW::Vec4f(orig_light, 1)).getXYZ();
		app.setLight(fake_light, lightc);
	}

	void Renderer::setLight(const math::float3& pos, const math::float3& color)
	{
		orig_light = FW::Vec3f(pos.x, pos.y, pos.z);
		lightc = FW::Vec3f(color.x, color.y, color.z);

		FW::Vec3f fake_light = (model.inverted() * FW::Vec4f(orig_light,1)).getXYZ();
		app.setLight(fake_light, FW::Vec3f(color.x, color.y, color.z));
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

