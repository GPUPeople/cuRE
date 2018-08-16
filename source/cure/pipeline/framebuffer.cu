


#include <math/vector.h>

#include "../shaders/tonemap.cuh"
#include "../shaders/uniforms.cuh"

#define FRAMEBUFFER_GLOBAL
#include "framebuffer.cuh"
#include "viewport.cuh"

__device__ void clamp_d(math::int2& v, const math::int2& lim)
{
	v.x = max(0, min(lim.x, v.x));
	v.y = max(0, min(lim.y, v.y));
}

__device__ void add(math::float4& sum, float weight, float4& color)
{
	sum.x += weight * color.x;
	sum.y += weight * color.y;
	sum.z += weight * color.z;
}

extern "C"
{
	__global__ void smoothTest1(cudaSurfaceObject_t target, cudaTextureObject_t source)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			math::float2 c;

			int left = 8 * (x / 8);
			int bottom = 8 * (y / 8);
			int right = left + 8;
			int up = bottom + 8;

			c.x = (left + right) * 0.5f * pixel_scale.x + pixel_scale.z;
			c.y = (bottom + up) * 0.5f * pixel_scale.y + pixel_scale.w;

			if (dot(c, c) > uniform[2])
			{
				math::float4 sum = { 0, 0, 0, 0 };
				math::float2 candidate;

				if (uniform[1] == 1)
				{
					math::uint2 in_rep(x % 4, y % 4);

					math::float2 close;
					close.x = (in_rep.x % 2 ? 1.f : -1.f);
					close.y = (in_rep.y % 2 ? 1.f : -1.f);

					if (in_rep.x < 2 ^ in_rep.y < 2) // in unfilled
					{
						math::float2 loc = { x + 0.5f, y + 0.5f };

						float4 sample;

						candidate.x = loc.x + close.x;
						candidate.y = loc.y;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.5f, sample);

						candidate.x = loc.x;
						candidate.y = loc.y + close.y;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.5f, sample);

						uchar4 color = { static_cast<unsigned char>(255 * sum.x), static_cast<unsigned char>(255 * sum.y), static_cast<unsigned char>(255 * sum.z), static_cast<unsigned char>(255 * sum.w) };

						FrameBuffer::writeColor(x, y, color);
					}
				}
			}
		}
	}

	__global__ void smoothTest2(cudaSurfaceObject_t target, cudaTextureObject_t source)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			math::float2 c;

			int left = 8 * (x / 8);
			int bottom = 8 * (y / 8);
			int right = left + 8;
			int up = bottom + 8;

			c.x = (left + right) * 0.5f * pixel_scale.x + pixel_scale.z;
			c.y = (bottom + up) * 0.5f * pixel_scale.y + pixel_scale.w;

			if (dot(c, c) > uniform[2])
			{
				math::float4 sum = { 0, 0, 0, 0 };
				math::float2 candidate;

				if (uniform[1] == 1)
				{
					math::uint2 in_rep(x % 4, y % 4);

					math::float2 close;
					close.x = (in_rep.x % 2 ? 1.f : -1.f);
					close.y = (in_rep.y % 2 ? 1.f : -1.f);

					if ((in_rep.x < 2 ^ in_rep.y >= 2)) // in filled
					{
						if (close.y < 0)
						{
							math::float2 loc = { x + 0.5f, y + 0.5f };

							float4 sample;

							candidate.x = loc.x;
							candidate.y = loc.y;
							sample = tex2D<float4>(source, candidate.x, candidate.y);
							add(sum, 0.5f, sample);

							candidate.x = loc.x + close.x;
							candidate.y = loc.y + close.y;
							sample = tex2D<float4>(source, candidate.x, candidate.y);
							add(sum, 0.5f, sample);

							uchar4 color = { static_cast<unsigned char>(255 * sum.x), static_cast<unsigned char>(255 * sum.y), static_cast<unsigned char>(255 * sum.z), static_cast<unsigned char>(255 * sum.w) };

							FrameBuffer::writeColor(x, y, color);
							FrameBuffer::writeColor(x + close.x, y + close.y, color);
						}
					}
				}
			}
		}
	}

	__global__ void smoothTest3(cudaSurfaceObject_t target, cudaTextureObject_t source)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			math::float2 c;

			int left = 8 * (x / 8);
			int bottom = 8 * (y / 8);
			int right = left + 8;
			int up = bottom + 8;

			c.x = (left + right) * 0.5f * pixel_scale.x + pixel_scale.z;
			c.y = (bottom + up) * 0.5f * pixel_scale.y + pixel_scale.w;

			if (dot(c, c) > uniform[2])
			{
				math::float4 sum = { 0, 0, 0, 0 };
				math::float2 candidate;

				if (uniform[1] == 1)
				{
					math::float2 loc = { x + 0.5f, y + 0.5f };

					float4 sample;

					math::float4 sum = { 0, 0, 0, 0 };

					candidate = { loc.x, loc.y - 1 };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.0625f, sample);

					candidate = { loc.x, loc.y - 1 };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.125f, sample);

					candidate = { loc.x + 1, loc.y - 1 };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.0625f, sample);

					//

					candidate = { loc.x - 1, loc.y };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.125f, sample);

					candidate = { loc.x, loc.y };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.25f, sample);

					candidate = { loc.x + 1, loc.y };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.125f, sample);

					//

					candidate = { loc.x - 1, loc.y + 1 };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.0625f, sample);

					candidate = { loc.x, loc.y + 1 };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.125f, sample);

					candidate = { loc.x + 1, loc.y + 1 };
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.0625f, sample);

					//

					uchar4 color = { static_cast<unsigned char>(255 * sum.x), static_cast<unsigned char>(255 * sum.y), static_cast<unsigned char>(255 * sum.z), static_cast<unsigned char>(255 * sum.w) };
					FrameBuffer::writeColor(x, y, color);
				}
			}
		}
	}

	__global__ void smoothImageQuad(cudaSurfaceObject_t target, cudaTextureObject_t source)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			int left = 8 * (x / 8);
			int bottom = 8 * (y / 8);

			math::float2 c { left + 4 - (viewport.left + viewport.right) * 0.5f, bottom + 4 - (viewport.top + viewport.bottom) * 0.5f };

			if (dot(c, c) > CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS)
			{
				math::float4 sum = { 0, 0, 0, 1 };
				math::float2 candidate;

				math::uint2 in_rep(x % 4, y % 4);

				math::float2 close((in_rep.x % 2 ? 1 : -1), (in_rep.y % 2 ? 1 : -1));

				if (in_rep.x < 2 ^ in_rep.y < 2) // in unfilled
				{
					math::float2 loc = { x + 0.5f, y + 0.5f };

					float4 sample;

					candidate.x = loc.x + close.x;
					candidate.y = loc.y - close.y * 0.25f;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.375f, sample);

					candidate.x = loc.x - close.x * 0.25f;
					candidate.y = loc.y + close.y;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.375f, sample);

					candidate.x = loc.x - close.x * 0.25f;
					candidate.y = loc.y - 2.f * close.y;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.125f, sample);

					candidate.x = loc.x - 2.f * close.x;
					candidate.y = loc.y - close.y * 0.25f;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.125f, sample); 

					FrameBuffer::writeColor(target, x, y, sum);
				}
				else
				{
					math::float2 loc = { x + 0.5f, y + 0.5f };

					float4 sample;

					candidate.x = loc.x - 0.25f * close.x;
					candidate.y = loc.y - 0.25f * close.y;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.5f, sample);

					candidate.x = loc.x + close.x;
					candidate.y = loc.y + close.y;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.28125f, sample);

					candidate.x = loc.x - 2 * close.x;
					candidate.y = loc.y + close.y;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.09375f, sample);

					candidate.x = loc.x + close.x;
					candidate.y = loc.y - 2 * close.y;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.09375f, sample);

					candidate.x = loc.x - 2 * close.x;
					candidate.y = loc.y - 2 * close.y;
					sample = tex2D<float4>(source, candidate.x, candidate.y);
					add(sum, 0.03125f, sample);

					FrameBuffer::writeColor(target, x, y, sum);
				}
			}
			else
			{
				math::float2 loc = { x + 0.5f, y + 0.5f };

				float4 sample = tex2D<float4>(source, loc.x, loc.y);

				FrameBuffer::writeColor(target, x, y, math::float4(sample.x, sample.y, sample.z, sample.w));
			}
		}
	}

	__device__ float inside(const math::int2& in_tile, float offx, float offy, math::float2 c)
	{
		math::float2 unit;
		unit.x = (in_tile.x + offx < 0 ? -1.f : in_tile.x + offx >= 8);
		unit.y = (in_tile.y + offy < 0 ? -1.f : in_tile.y + offy >= 8);
		c.x += unit.x * 8;
		c.y += unit.y * 8;
		return (dot(c, c) < CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS ? 1.f : 0.0f);
	}

	__global__ void smoothImage(cudaSurfaceObject_t target, cudaTextureObject_t source)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			int left = 8 * (x / 8);
			int bottom = 8 * (y / 8);

			math::float2 c { left + 4 - (viewport.left + viewport.right) * 0.5f, bottom + 4 - (viewport.top + viewport.bottom) * 0.5f };

			if (dot(c, c) > CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS)
			{
				math::float4 sum = { 0, 0, 0, 1 };
				math::float2 candidate;

				math::float2 c2 { abs(c.x) - 8, abs(c.y) - 8 };

				if (dot(c2, c2) > CHECKERBOARD_RADIUS * CHECKERBOARD_RADIUS)
				{
					if ((x % 2) ^ (y % 2)) // in unfilled
					{
						math::float2 loc(x, y);

						float4 sample;

						candidate.x = loc.x;
						candidate.y = loc.y;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.875f, sample);

						candidate.x = loc.x + 1;
						candidate.y = loc.y + 1;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.875f, sample);

						candidate.x = loc.x - 1;
						candidate.y = loc.y - 1;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.0625F, sample);

						candidate.x = loc.x + 2;
						candidate.y = loc.y - 1;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.0625F, sample);

						candidate.x = loc.x + 2;
						candidate.y = loc.y + 2;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.0625F, sample);

						candidate.x = loc.x - 1;
						candidate.y = loc.y + 2;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.0625F, sample);

						FrameBuffer::writeColor(target, x, y, sum);
					}
					else
					{
						math::float2 loc = { x + 0.5f, y + 0.5f };

						float4 sample;

						float third = 1.f / 3.f;

						candidate.x = loc.x;
						candidate.y = loc.y;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.375f, sample);

						candidate.x = loc.x - 1 + third;
						candidate.y = loc.y - 1 - third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.28125f, sample);

						candidate.x = loc.x + 1 + third;
						candidate.y = loc.y - 1 + third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.28125f, sample);

						candidate.x = loc.x + 1 - third;
						candidate.y = loc.y + 1 + third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.28125f, sample);

						candidate.x = loc.x - 1 - third;
						candidate.y = loc.y + 1 - third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.28125f, sample);

						FrameBuffer::writeColor(target, x, y, sum);
					}
				}
				else
				{
					math::int2 in_tile = { x - left, y - bottom };

					float4 sample;

					float in, in2;

					if ((x % 2) ^ (y % 2)) // in unfilled
					{
						math::float2 loc(x, y);

						candidate.x = loc.x;
						candidate.y = loc.y;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in = inside(in_tile, -1, -1, c);
						add(sum, 0.875f - (in * 0.2916666f), sample);

						candidate.x = loc.x - 1;
						candidate.y = loc.y - 1;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in2 = inside(in_tile, -2, -2, c);
						add(sum, 0.0625F - (in2 * 0.02083333f) - (in * 0.010416666f), sample);

						candidate.x = loc.x + 1;
						candidate.y = loc.y + 1;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						
						in = inside(in_tile, 1, 1, c);
						add(sum, 0.875f - (in * 0.2916666f), sample);

						candidate.x = loc.x + 2;
						candidate.y = loc.y + 2;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in2 = inside(in_tile, 2, 2, c);
						add(sum, 0.0625F - (in2 * 0.02083333f) - (in * 0.010416666f), sample);

						candidate.x = loc.x + 2;
						candidate.y = loc.y - 1;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in = inside(in_tile, 1, -1, c);
						in2 = inside(in_tile, 2, -2, c);
						add(sum, 0.0625F - (in2 * 0.02083333f) - (in * 0.010416666f), sample);

						candidate.x = loc.x - 1;
						candidate.y = loc.y + 2;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in = inside(in_tile, -1, 1, c);
						in2 = inside(in_tile, -2, 2, c);
						add(sum, 0.0625F - (in2 * 0.02083333f) - (in * 0.010416666f), sample);

						FrameBuffer::writeColor(target, x, y, sum);
					}
					else
					{
						math::float2 loc = { x + 0.5f, y + 0.5f };

						float third = 1.f / 3.f;

						candidate.x = loc.x;
						candidate.y = loc.y;
						sample = tex2D<float4>(source, candidate.x, candidate.y);
						add(sum, 0.375f, sample);

						candidate.x = loc.x - 1 + third;
						candidate.y = loc.y - 1 - third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in = inside(in_tile, 0, -1, c);
						in2 = inside(in_tile, -1, -2, c);
						add(sum, 0.28125f - in2 * 0.1026785f - in * 0.022321f, sample);

						candidate.x = loc.x + 1 + third;
						candidate.y = loc.y - 1 + third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in = inside(in_tile, 1, 0, c);
						in2 = inside(in_tile, 2, -1, c);
						add(sum, 0.28125f - in2 * 0.1026785f - in * 0.022321f, sample);

						candidate.x = loc.x + 1 - third;
						candidate.y = loc.y + 1 + third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in = inside(in_tile, 0, 1, c);
						in2 = inside(in_tile, 1, 2, c);
						add(sum, 0.28125f - in2 * 0.1026785f - in * 0.022321f, sample);

						candidate.x = loc.x - 1 - third;
						candidate.y = loc.y + 1 - third;
						sample = tex2D<float4>(source, candidate.x, candidate.y);

						in = inside(in_tile, -1, 0, c);
						in2 = inside(in_tile, -2, 1, c);
						add(sum, 0.28125f - in2 * 0.1026785f - in * 0.022321f, sample);

						FrameBuffer::writeColor(target, x, y, sum);
					}
				}
			}
			else
			{
				math::float2 loc = { x + 0.5f, y + 0.5f };

				float4 sample = tex2D<float4>(source, loc.x, loc.y);

				FrameBuffer::writeColor(target, x, y, math::float4(sample.x, sample.y, sample.z, sample.w));
			}
		}
	}

	__global__ void clearColorBuffer(uchar4 color)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			FrameBuffer::writeColor(x, y, color);
		}
	}

	__global__ void clearColorBufferCheckers(uchar4 c1, uchar4 c2, unsigned int s)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			FrameBuffer::writeColor(x, y, (((x >> s) ^ (y >> s)) & 0x1U) == 0U ? c1 : c2);
		}
	}

	__global__ void clearColorBufferTexture()
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < color_buffer_size.x && y < color_buffer_size.y)
		{
			math::float2 px = { 2.f * ((x + 0.5f) / ((float)color_buffer_size.x)) - 1.f,
									 2.f * ((y + 0.5f) / ((float)color_buffer_size.y)) - 1.f };

			math::float4 on_near = { px.x, px.y, 0, 1.f };
			math::float4 on_far = { px.x, px.y, 1, 1.f };

			on_near = camera.PV_inv * on_near;
			on_far = camera.PV_inv * on_far;

			math::float3 r = normalize(on_far.xyz() / on_far.w - on_near.xyz() / on_near.w);

			math::float2 uv;
			uv.x = (1.f / (2.f * math::constants<float>::pi())) * (atan2(r.z, r.x) + math::constants<float>::pi());
			uv.y = 1.f - (2.f / (math::constants<float>::pi()) * asin(r.y));

			float4 tex_color_ = tex2D(texf, uv.x, uv.y);

			math::float3 tex_color(tex_color_.x, tex_color_.y, tex_color_.z);
			tex_color = tonemap(tex_color);

			//uchar4 color = { fminf(1.f, tex_color.x) * 255, fminf(1.f, tex_color.y) * 255, fminf(1.f, tex_color.z) * 255, 255 };

			FrameBuffer::writeColor(x, y, { tex_color, 1.0f });
		}
	}

	__global__ void clearDepthBuffer(float depth)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < depth_buffer_size.x && y < depth_buffer_size.y)
		{
			FrameBuffer::writeDepth(x, y, depth);
		}
	}
}
