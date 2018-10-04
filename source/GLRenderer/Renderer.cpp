


#include <stdexcept>
#include <iostream>

#include <ResourceImp.h>

#include "config.h"

#include "Geometry.h"
#include "Texture.h"
#include "materials/ColoredMaterial.h"
#include "materials/TexturedMaterial.h"
#include "materials/LitMaterial.h"
#include "materials/TexturedLitMaterial.h"
#include "materials/ClipspaceMaterial.h"
#include "materials/HeavyMaterials.h"
#include "materials/EyeCandyMaterial.h"
#include "Renderer.h"


namespace GLRenderer
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

	void PerformanceMonitor::recordDrawingTime(double t) const
	{
		if (callback)
			callback->recordDrawingTime(t);
	}

	void PerformanceMonitor::recordMemoryStatus() const
	{
		if (callback)
		{
			GLint64 total;
			glGetInteger64v(0x9047, &total);

			GLint64 free;
			glGetInteger64v(0x9049, &free);

			callback->recordMemoryStatus(free * 1024U, total * 1024U);
		}
	}


	Renderer::Renderer(PerformanceDataCallback* performance_callback)
		: perf_mon(performance_callback)
	{
		std::cout << "OpenGL renderer using " << glGetString(GL_RENDERER) << std::endl;
	}

	::Geometry* Renderer::createIndexedTriangles(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	{
		auto g = ResourceImp<IndexedTriangles>::create(*this, position, normals, texcoord, num_vertices, indices, num_indices);
		perf_mon.recordMemoryStatus();
		return g;
	}

	::Geometry* Renderer::createIndexedQuads(const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	{
		auto g = ResourceImp<IndexedQuads>::create(*this, position, normals, texcoord, num_vertices, indices, num_indices);
		perf_mon.recordMemoryStatus();
		return g;
	}

	::Geometry* Renderer::createClipspaceGeometry(const float* position, size_t num_vertices)
	{
		auto g = ResourceImp<ClipspaceGeometry>::create(*this, position, num_vertices);
		perf_mon.recordMemoryStatus();
		return g;
	}

	::Texture* Renderer::createTexture2DRGBA8(size_t width, size_t height, unsigned int levels, const std::uint32_t* data)
	{
		auto t = ResourceImp<Texture2DRGBA8>::create(*this, static_cast<GLsizei>(width), static_cast<GLsizei>(height), static_cast<GLsizei>(levels), data);
		perf_mon.recordMemoryStatus();
		return t;
	}

	::Material* Renderer::createColoredMaterial(const math::float4& color)
	{
		auto m = ResourceImp<ColoredMaterial>::create(colored_shader, color);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createLitMaterial(const math::float4& color)
	{
		auto m = ResourceImp<LitMaterial>::create(lit_shader, color);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createVertexHeavyMaterial(int iterations)
	{
		auto m = ResourceImp<VertexHeavyMaterial>::create(heavy_vertex_shader, iterations);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createFragmentHeavyMaterial(int iterations)
	{
		auto m = ResourceImp<FragmentHeavyMaterial>::create(heavy_fragment_shader, iterations);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createTexturedMaterial(GLuint tex, const math::float4& color)
	{
		auto m = ResourceImp<TexturedMaterial>::create(textured_shader, tex, color);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createTexturedLitMaterial(GLuint tex, const math::float4& color)
	{
		auto m = ResourceImp<TexturedLitMaterial>::create(textured_lit_shader, tex, color);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createClipspaceMaterial()
	{
		auto m = ResourceImp<ClipspaceMaterial>::create(clipspace_shader);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createVertexHeavyClipspaceMaterial(int iterations)
	{
		auto m = ResourceImp<VertexHeavyClipspaceMaterial>::create(vertex_heavy_clipspace_shader, iterations);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createFragmentHeavyClipspaceMaterial(int iterations)
	{
		auto m = ResourceImp<FragmentHeavyClipspaceMaterial>::create(fragment_heavy_clipspace_shader, iterations);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createEyeCandyMaterial()
	{
		auto m = ResourceImp<EyeCandyMaterial>::create(eyecandy_shader);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createVertexHeavyEyeCandyMaterial(int iterations)
	{
		auto m = ResourceImp<VertexHeavyEyeCandyMaterial>::create(vertex_heavy_eyecandy_shader, iterations);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Material* Renderer::createFragmentHeavyEyeCandyMaterial(int iterations)
	{
		auto m = ResourceImp<FragmentHeavyEyeCandyMaterial>::create(fragment_heavy_eyecandy_shader, iterations);
		perf_mon.recordMemoryStatus();
		return m;
	}

	::Geometry* Renderer::createEyeCandyGeometry(const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	{
		auto g = ResourceImp<EyeCandyGeometry>::create(*this, position, num_vertices, indices, triangle_colors, num_triangles);
		perf_mon.recordMemoryStatus();
		return g;
	}


	void Renderer::setRenderTarget(GLuint color_buffer, int width, int height)
	{
		buffer_width = width;
		buffer_height = height;
		depth_buffer = GL::createRenderbuffer(width, height, GL_DEPTH_COMPONENT32);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		if (FRAGMENT_SHADER_INTERLOCK)
			this->color_buffer = color_buffer;
		else
			glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_buffer, 0);
		glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	}

	RendereringContext* Renderer::beginFrame()
	{
		shader_state.bind();

		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		if (DEPTH_TEST)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);

		glDepthMask(DEPTH_WRITE);

		if (BLENDING)
			glEnable(GL_BLEND);
		else
			glDisable(GL_BLEND);

		if (BACKFACE_CULLING != 0)
			glEnable(GL_CULL_FACE);
		else
			glDisable(GL_CULL_FACE);

		if (BACKFACE_CULLING < 0)
			glFrontFace(GL_CCW);
		else
			glFrontFace(GL_CW);

		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		if (FRAGMENT_SHADER_INTERLOCK)
			glBindImageTexture(0, color_buffer, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
			//glBindImageTexture(0, color_buffer, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
			//glBindImageTexture(0, color_buffer, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

		perf_mon.recordMemoryStatus();

		drawing_time = 0.0;

		return this;
	}

	void Renderer::finish()
	{
		glFrontFace(GL_CW);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		perf_mon.recordDrawingTime(drawing_time);
	}


	void Renderer::clearColorBuffer(float r, float g, float b, float a)
	{
		glClearColor(r, g, b, a);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void Renderer::clearColorBufferCheckers(std::uint32_t a, std::uint32_t b, unsigned int s)
	{
		throw std::runtime_error("not implemented here");
	}

	void Renderer::clearDepthBuffer(float depth)
	{
		glClearDepthf(depth);
		glClear(GL_DEPTH_BUFFER_BIT);
	}

	void Renderer::setViewport(float x, float y, float width, float height)
	{
		glViewportIndexedf(0U, x, y, width, height);
	}

	void Renderer::setUniformf(int index, float v)
	{
		//throw std::runtime_error("not implemented here");
	}

	void Renderer::setCamera(const Camera::UniformBuffer& params)
	{
		shader_state.setCamera(params);
	}

	void Renderer::setObjectTransform(const math::affine_float4x4& M)
	{
		shader_state.setObjectTransform(M);
	}

	void Renderer::setLight(const math::float3& pos, const math::float3& color)
	{
		shader_state.setLight(pos, color);
	}


	void Renderer::beginDrawTiming()
	{
		glBeginQuery(GL_TIME_ELAPSED, drawing_time_query);
	}

	void Renderer::endDrawTiming()
	{
		glEndQuery(GL_TIME_ELAPSED);

		GLuint64 time_elapsed;
		glGetQueryObjectui64v(drawing_time_query, GL_QUERY_RESULT, &time_elapsed);

		drawing_time += time_elapsed * 1e-9f;
	}


	::Renderer* __stdcall Renderer::create(int, PerformanceDataCallback* performance_callback)
	{
		return new Renderer(performance_callback);
	}

	void Renderer::destroy()
	{
		delete this;
	}
}
