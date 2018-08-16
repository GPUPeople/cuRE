


#include <limits>

#include <CUDA/error.h>

#include <math/math.h>

#include "utils.h"

#include "Framebuffer.h"
#include "PipelineModule.h"

namespace
{
	static std::uint8_t toLinear8(float c)
	{
		return static_cast<std::uint8_t>(math::saturate(c) * 255.0f);
	}

	static std::uint8_t toSRGB8(float c)
	{
		return toLinear8(pow(c, 1.0f / 2.2f));
	}

	std::uint32_t sRGB8_A8(float r, float g, float b, float a = 1.0f)
	{
		return (toLinear8(a) << 24U) | (toSRGB8(b) << 16U) | (toSRGB8(g) << 8U) | toSRGB8(r);
	}
}

namespace cuRE
{
	Framebuffer::Framebuffer(const PipelineModule& module)
	    : color_buffer(module.getSurfaceReference("color_buffer")),
	      color_buffer_size(module.getGlobal("color_buffer_size")),
	      depth_buffer(module.getSurfaceReference("depth_buffer")),
	      depth_buffer_size(module.getGlobal("depth_buffer_size")),
	      clear_color_buffer(module.getFunction("clearColorBuffer")),
	      clear_color_buffer_checkers(module.getFunction("clearColorBufferCheckers")),
	      clear_color_buffer_texture(module.getFunction("clearColorBufferTexture")),
	      clear_depth_buffer(module.getFunction("clearDepthBuffer")),
	      smooth_image_quad(module.getFunction("smoothImageQuad")),
	      smooth_image(module.getFunction("smoothImage")),
	      test_image(module.getFunction("smoothTest1")),
	      test_image2(module.getFunction("smoothTest2")),
	      test_image3(module.getFunction("smoothTest3"))
	{
	}

	void Framebuffer::attachColorBuffer(CUarray buffer, unsigned int width, unsigned int height)
	{
		color_buffer_width = width;
		color_buffer_height = height;

		succeed(cuSurfRefSetArray(color_buffer, buffer, 0U));
		unsigned int size[] = {width, height};
		succeed(cuMemcpyHtoD(color_buffer_size, size, 2 * 4U));


		CUDA_RESOURCE_DESC resource_desc;
		resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
		resource_desc.res.array.hArray = buffer;
		resource_desc.flags = 0U;

		CUDA_TEXTURE_DESC texture_desc = {
			{ CU_TR_ADDRESS_MODE_MIRROR, CU_TR_ADDRESS_MODE_MIRROR , CU_TR_ADDRESS_MODE_MIRROR },  // addressMode
			CU_TR_FILTER_MODE_LINEAR,                                                              // filterMode
			CU_TRSF_SRGB,                                                                          // flags
			1U,                                                                                    // maxAnisotropy
			CU_TR_FILTER_MODE_LINEAR,                                                              // mipmapFilterMode
			0.0f,                                                                                  // mipmapLevelBias
			0.0f,                                                                                  // minMipmapLevelClamp
			std::numeric_limits<float>::max(),                                                     // maxMipmapLevelClamp
			{ 0.0f, 0.0f, 0.0f, 0.0f }                                                             // borderColor
		};

		color_as_texture.reset();
		color_as_texture = CU::createTextureObject(resource_desc, texture_desc);
	}

	void Framebuffer::attachDepthBuffer(CUarray depth_buffer, unsigned int width, unsigned int height)
	{
		depth_buffer_width = width;
		depth_buffer_height = height;

		succeed(cuSurfRefSetArray(this->depth_buffer, depth_buffer, 0U));

		//succeed(cuMemcpyHtoD(this->depth_buffer, &depth_buffer, sizeof(CUdeviceptr)));
		unsigned int size[] = {width, height};
		succeed(cuMemcpyHtoD(depth_buffer_size, size, 2 * 4U));
	}

	void Framebuffer::clearColorBuffer(float r, float g, float b, float a)
	{
		const unsigned int block_size_x = 16;
		const unsigned int block_size_y = 16;
		unsigned int num_blocks_x = divup(color_buffer_width, block_size_x);
		unsigned int num_blocks_y = divup(color_buffer_height, block_size_y);

		clear_color_buffer({num_blocks_x, num_blocks_y}, {block_size_x, block_size_y}, 0U, nullptr, sRGB8_A8(r, g, b, a));
		succeed(cuCtxSynchronize());
	}

	void Framebuffer::upsample(CUsurfObject target)
	{
		const unsigned int block_size_x = 16;
		const unsigned int block_size_y = 16;
		unsigned int num_blocks_x = divup(color_buffer_width, block_size_x);
		unsigned int num_blocks_y = divup(color_buffer_height, block_size_y);

		smooth_image({ num_blocks_x, num_blocks_y }, { block_size_x, block_size_y }, 0U, nullptr, target, color_as_texture);
	}

	void Framebuffer::upsampleQuad(CUsurfObject target)
	{
		const unsigned int block_size_x = 16;
		const unsigned int block_size_y = 16;
		unsigned int num_blocks_x = divup(color_buffer_width, block_size_x);
		unsigned int num_blocks_y = divup(color_buffer_height, block_size_y);

		smooth_image_quad({num_blocks_x, num_blocks_y}, {block_size_x, block_size_y}, 0U, nullptr, target, color_as_texture);
	}

	//void Framebuffer::upsample(CUsurfObject target)
	//{
	//	const unsigned int block_size_x = 16;
	//	const unsigned int block_size_y = 16;
	//	unsigned int num_blocks_x = divup(color_buffer_width, block_size_x);
	//	unsigned int num_blocks_y = divup(color_buffer_height, block_size_y);

	//	smooth_image({ num_blocks_x, num_blocks_y }, { block_size_x, block_size_y }, 0U, nullptr, target, color_as_texture);
	//}

	void Framebuffer::clearColorBufferCheckers(std::uint32_t c1, std::uint32_t c2, unsigned int s)
	{
		const unsigned int block_size_x = 16;
		const unsigned int block_size_y = 16;
		unsigned int num_blocks_x = divup(color_buffer_width, block_size_x);
		unsigned int num_blocks_y = divup(color_buffer_height, block_size_y);

		clear_color_buffer_checkers({num_blocks_x, num_blocks_y}, {block_size_x, block_size_y}, 0U, nullptr, c1, c2, s);
		succeed(cuCtxSynchronize());
	}

	void Framebuffer::clearColorBufferToTextureF()
	{
		const unsigned int block_size_x = 16;
		const unsigned int block_size_y = 16;
		unsigned int num_blocks_x = divup(color_buffer_width, block_size_x);
		unsigned int num_blocks_y = divup(color_buffer_height, block_size_y);

		clear_color_buffer_texture({num_blocks_x, num_blocks_y}, {block_size_x, block_size_y}, 0U, nullptr);
		succeed(cuCtxSynchronize());
	}

	void Framebuffer::clearDepthBuffer(float depth)
	{
		const unsigned int block_size_x = 16;
		const unsigned int block_size_y = 16;
		unsigned int num_blocks_x = divup(depth_buffer_width, block_size_x);
		unsigned int num_blocks_y = divup(depth_buffer_height, block_size_y);

		clear_depth_buffer({num_blocks_x, num_blocks_y}, {block_size_x, block_size_y}, 0U, nullptr, depth);
		succeed(cuCtxSynchronize());
	}
}
