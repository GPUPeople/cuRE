


#ifndef INCLUDED_CURE_PIPELINE_FRAMEBUFFER
#define INCLUDED_CURE_PIPELINE_FRAMEBUFFER

#pragma once

#include <cstdint>

#include <CUDA/module.h>
#include <CUDA/texture.h>
#include <CUDA/launch.h>


namespace cuRE
{
	class PipelineModule;

	class Framebuffer
	{
	private:
		CUsurfref color_buffer;
		CUdeviceptr color_buffer_size;

		unsigned int color_buffer_width;
		unsigned int color_buffer_height;

		CUsurfref depth_buffer;
		//CUdeviceptr depth_buffer;
		CUdeviceptr depth_buffer_size;

		unsigned int depth_buffer_width;
		unsigned int depth_buffer_height;

		CU::unique_texture color_as_texture;

		CU::Function<std::uint32_t> clear_color_buffer;
		CU::Function<std::uint32_t, std::uint32_t, std::uint32_t> clear_color_buffer_checkers;
		CU::Function<> clear_color_buffer_texture;
		CU::Function<CUsurfObject, CUtexObject> smooth_image_quad;
		CU::Function<CUsurfObject, CUtexObject> smooth_image;

		CU::Function<CUsurfObject, CUtexObject> test_image;
		CU::Function<CUsurfObject, CUtexObject> test_image2;
		CU::Function<CUsurfObject, CUtexObject> test_image3;

		CU::Function<float> clear_depth_buffer;

	public:
		Framebuffer(const Framebuffer&) = delete;
		Framebuffer& operator =(const Framebuffer&) = delete;

		Framebuffer(const PipelineModule& module);

		void attachColorBuffer(CUarray color_buffer, unsigned int width, unsigned int height);
		void attachDepthBuffer(CUarray depth_buffer, unsigned int width, unsigned int height);

		void clearColorBuffer(float r, float g, float b, float a);
		void clearColorBufferCheckers(std::uint32_t c1, std::uint32_t c2, unsigned int s);
		void clearColorBufferToTextureF();
		void clearDepthBuffer(float depth);

		void upsample(CUsurfObject target);
		void upsampleQuad(CUsurfObject target);
	};
}

#endif  // INCLUDED_CURE_PIPELINE_FRAMEBUFFER
