


#ifndef INCLUDED_FREEPIPE_FRAGMENT_PROCESSING
#define INCLUDED_FREEPIPE_FRAGMENT_PROCESSING

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "config.h"
#include "fragment_data.cuh"
#include "IntermediateGeometryStorage.cuh"

#include "shaders/vertex_simple.cuh"
#include "shaders/clipspace.cuh"


#include "ptx_primitives.cuh"

extern "C"
{
	struct Viewport
	{
		float left;
		float top;
		float right;
		float bottom;
	};
	__constant__ Viewport viewport;

	__constant__ float pixel_step[2];

	__constant__ float* c_depthBuffer;
	__constant__ unsigned int c_bufferDims[2];

	surface<void, cudaSurfaceType2D> color_buffer;

	__global__ void clearColorBuffer(uchar4 color, unsigned int buffer_width, unsigned int buffer_height)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < buffer_width && y < buffer_height)
		{
			surf2Dwrite(color, color_buffer, 4 * x, y);
		}
	}
}
extern "C"
{
	__constant__ float *c_positions, *c_normals, *c_texCoords;
	__constant__ unsigned int *c_indices, *c_patchData;

}

namespace FreePipe
{
	template<class FragmentIn, int TInterpolators>
	struct Interpolators
	{
		static const int NumInterpolators = FragmentIn::Interpolators;
		math::float3 interpolators[NumInterpolators];
		__device__ void setup(const FragmentIn* v0, const FragmentIn* v1, const FragmentIn* v2, const math::float3x3& M_inv)
		{
			#pragma unroll
			for (int i = 0; i < NumInterpolators; ++i)
				interpolators[i] = math::float3(v0->attributes[i], v1->attributes[i], v2->attributes[i]) * M_inv;
		}

		__device__ void interpolate(FragmentIn& frag, math::float3 p, float w)
		{
			#pragma unroll
			for (int i = 0; i < NumInterpolators; ++i)
				frag.attributes[i] = dot(interpolators[i], p) * w;
		}
	};

	template<class FragmentIn>
	struct Interpolators<FragmentIn, 0>
	{
		__device__ void setup(const FragmentIn* v0, const FragmentIn* v1, const FragmentIn* v2, const math::float3x3& M_inv)
		{
		}

		__device__ void interpolate(FragmentIn& frag, math::float3 p, float w)
		{
		}
	};

	

	template<class VertexShader>
	__device__ typename VertexShader::VertexOut runVertexShader(unsigned int vId)
	{
		math::float3 pos = math::float3(c_positions[vId * 3], c_positions[vId * 3 + 1], c_positions[vId * 3 + 2]);
		math::float3 normal = math::float3(c_normals[vId * 3], c_normals[vId * 3 + 1], c_normals[vId * 3 + 2]);
		math::float2 tex = math::float2(c_texCoords[vId * 2], c_texCoords[vId * 2 + 1]);
		typename VertexShader::VertexOut out = VertexShader::process(pos, normal, tex);
		return out;
	}

	template<>
	__device__ typename Shaders::ClipSpaceVertexShader::VertexOut runVertexShader<Shaders::ClipSpaceVertexShader>(unsigned int vId)
	{
		math::float4 pos = math::float4(c_positions[vId * 4], c_positions[vId * 4 + 1], c_positions[vId * 4 + 2], c_positions[vId * 4 + 3]);
		return Shaders::ClipSpaceVertexShader::process(pos);
	}


	template<class VertexShader, class FragmentShader>
	__device__ void process_fragments_h(float* depth_buffer, unsigned int buffer_width, unsigned int buffer_height, float pixel_step_x, float pixel_step_y, unsigned int triangle_id)
	{

		unsigned int id = triangle_id;

		// every thread works on one triangle for now
		//if (id < num_triangles)
		{
			typename VertexShader::VertexOut vert0 = runVertexShader<VertexShader>(c_indices[3 * id]);
			typename VertexShader::VertexOut vert1 = runVertexShader<VertexShader>(c_indices[3 * id + 1]);
			typename VertexShader::VertexOut vert2 = runVertexShader<VertexShader>(c_indices[3 * id + 2]);

			//unsigned int id0 = geometryOutStorage.indices[3 * id];
			//unsigned int id1 = geometryOutStorage.indices[3 * id + 1];
			//unsigned int id2 = geometryOutStorage.indices[3 * id + 2];

			math::float4 v0 = vert0.pos;
			math::float4 v1 = vert1.pos;
			math::float4 v2 = vert2.pos;


			math::float3 p0 = math::float3(v0.x, v0.y, v0.w);
			math::float3 p1 = math::float3(v1.x, v1.y, v1.w);
			math::float3 p2 = math::float3(v2.x, v2.y, v2.w);

			math::float3x3 M = math::float3x3(
				v0.x, v1.x, v2.x,
				v0.y, v1.y, v2.y,
				v0.w, v1.w, v2.w
			);

			math::float3x3 M_adj = adj(M);

			float det = dot(M_adj.row1(), M.column1());

			if (det > -0.00001f)
				return;

			math::float3x3 M_inv = (1.0f / det) * M_adj;

			float l0 = 1.0f / v0.w;
			float l1 = 1.0f / v1.w;
			float l2 = 1.0f / v2.w;

			math::float3 u0 = M_inv.row1();
			math::float3 u1 = M_inv.row2();
			math::float3 u2 = M_inv.row3();


			math::float3 uw = math::float3(1.0f, 1.0f, 1.0f) * M_inv;
			math::float3 uz = math::float3(v0.z, v1.z, v2.z) * M_inv;

			Interpolators<typename VertexShader::VertexOut, VertexShader::VertexOut::Interpolators> interpolator;
			interpolator.setup(&vert0, &vert1, &vert2, M_inv);


			float vp_scale_x = 0.5f * (viewport.right - viewport.left);
			float vp_scale_y = 0.5f * (viewport.bottom - viewport.top);

			float x0 = (v0.x * l0 + 1.0f) * vp_scale_x + viewport.left;
			float x1 = (v1.x * l1 + 1.0f) * vp_scale_x + viewport.left;
			float x2 = (v2.x * l2 + 1.0f) * vp_scale_x + viewport.left;

			float y0 = (v0.y * l0 + 1.0f) * vp_scale_y + viewport.top;
			float y1 = (v1.y * l1 + 1.0f) * vp_scale_y + viewport.top;
			float y2 = (v2.y * l2 + 1.0f) * vp_scale_y + viewport.top;

			float x_min = max(min(x0, min(x1, x2)), viewport.left);
			float y_min = max(min(y0, min(y1, y2)), viewport.top);

			float x_max = min(max(x0, max(x1, x2)), viewport.right);
			float y_max = min(max(y0, max(y1, y2)), viewport.bottom);

			int i_start = ceil(x_min - 0.50000f);
			int j_start = ceil(y_min - 0.50000f);

			int i_end = x_max + 0.50000f;
			int j_end = y_max + 0.50000f;

			//printf("%d %d %d %d\n", i_start, i_end, j_start, j_end);

			for (int j = j_start; j < j_end; ++j)
			{
				float y = -1.0f + (j + 0.5f) * pixel_step[1];

				for (int i = i_start; i < i_end; ++i)
				{
					float x = -1.0f + (i + 0.5f) * pixel_step[0];

					math::float3 p = math::float3(x, y, 1.0f);

					float f0 = dot(u0, p);
					float f1 = dot(u1, p);
					float f2 = dot(u2, p);

					if (f0 >= 0.0f && f1 >= 0.0f && f2 >= 0.0f)
					{
						float rcpw = dot(uw, p);
						float w = 1.0f / rcpw;

						float z = dot(uz, p);

						typename FragmentShader::FragementIn frag;
						frag.pos = math::float4(x, y, z, 1.0f);



						interpolator.interpolate(frag, p, w);

						FragementData data;
						data.depth = z;

						if (z >= -1.0f && z <= 1.0f)  // clipping!
						{
							if (FragmentShader::process(data, frag))
							{

								//float c = 0.5f*(z + 1.0f);
								//float c = f0 * w;


								//float* pd = depth_buffer + j * buffer_width + i;

								//float current_z;
								//bool notdone = true;
								//while (notdone)
								//{
								//	current_z = atomicExch(pd, -1.0f);

								//	if (current_z != -1.0f)
								//	{

								//		if (data.depth < current_z)
								//		{
								//			surf2Dwrite(make_uchar4(255 * data.color.x, 255 * data.color.y, 255 * data.color.z, 255 * data.color.w), color_buffer, 4 * i, j);
								//			__threadfence();
								//			//atomicExch(pd, z);
								//			*pd = data.depth;
								//		}
								//		else
								//			//atomicExch(pd, current_z);
								//			*pd = current_z;
								//		notdone = false;
								//	}
								//}


								float d = __float_as_int(data.depth);

								if (!DEPTH_TEST || d < atomicMin(reinterpret_cast<int*>(depth_buffer + j * buffer_width + i), d))
								{
									surf2Dwrite(make_uchar4(255 * data.color.x, 255 * data.color.y, 255 * data.color.z, 255), color_buffer, 4 * i, j);
								}

							}
						}
					}
				}
			}

		}
	}



	//template<class VertexShader, class FragmentShader>
	//__device__ void process_fragments(IntermediateGeometryStorage& geometryOutStorage, float* depth_buffer, 
	//                                  unsigned int buffer_width, unsigned int buffer_height, 
	//                                  float xPixelStep, float yPixelStep,
	//                                  unsigned int numTriangles, unsigned int triangleId)
	//{
	//	unsigned int id = triangleId;

	//	// every thread works on one triangle for now
	//	if (id < numTriangles)
	//	{
	//		unsigned int id0 = geometryOutStorage.indices[3 * id];
	//		unsigned int id1 = geometryOutStorage.indices[3 * id + 1];
	//		unsigned int id2 = geometryOutStorage.indices[3 * id + 2];

	//		math::float4 v0 = geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[id0].pos;
	//		math::float4 v1 = geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[id1].pos;
	//		math::float4 v2 = geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[id2].pos;

	//		// devide by w
	//		float w0_inv = 1.0f / v0.w;
	//		v0.x *= w0_inv; v0.y *= w0_inv; v0.z *= w0_inv;
	//		float w1_inv = 1.0f / v1.w;
	//		v1.x *= w1_inv; v1.y *= w1_inv; v1.z *= w1_inv;
	//		float w2_inv = 1.0f / v2.w;
	//		v2.x *= w2_inv; v2.y *= w2_inv; v2.z *= w2_inv;

	//		// compute edge equations
	//		math::float2 n0(-(v1.y - v0.y), v1.x - v0.x);
	//		math::float2 n1(-(v2.y - v1.y), v2.x - v1.x);
	//		math::float2 n2(-(v0.y - v2.y), v0.x - v2.x);
	//		float c0 = -dot(n0, math::float2(v0.x, v0.y));
	//		float c1 = -dot(n1, math::float2(v1.x, v1.y));
	//		float c2 = -dot(n2, math::float2(v2.x, v2.y));

	//		// compute bounding box limited to -1 +1
	//		float xmin = max(min(min(v0.x, v1.x), v2.x) - 0.00001f, -1.0f);
	//		float xmax = min(max(max(v0.x, v1.x), v2.x) + 0.00001f,  1.0f);
	//		float ymin = max(min(min(v0.y, v1.y), v2.y) - 0.00001f, -1.0f);
	//		float ymax = min(max(max(v0.y, v1.y), v2.y) + 0.00001f,  1.0f);

	//		// compute pixel positions
	//		float xstart = (viewport.left + 0.5f*(xmin + 1.0f)*(viewport.right - viewport.left));
	//		float xend = (viewport.left + 0.5f*(xmax + 1.0f)*(viewport.right - viewport.left));
	//		float ystart = (viewport.top + 0.5f*(ymin + 1.0f)*(viewport.bottom - viewport.top));
	//		float yend = (viewport.top + 0.5f*(ymax + 1.0f)*(viewport.bottom - viewport.top));

	//		//float xstart = 0;
	//		//float xend = buffer_width - 1;
	//		//float ystart = 0;
	//		//float yend = buffer_height - 1;

	//		unsigned int pxstart = xstart + 0.5f;
	//		unsigned int pystart = ystart + 0.5f;
	//		unsigned int pxend = min(static_cast<unsigned int>(xend), c_bufferDims[0] - 1);
	//		unsigned int pyend = min(static_cast<unsigned int>(yend), c_bufferDims[1] - 1);

	//		xstart = ((pxstart + 0.5f) - viewport.left) * xPixelStep - 1.0f;
	//		ystart = ((pystart + 0.5f) - viewport.top) * yPixelStep - 1.0f;

	//		// run through pixels inside bounding box
	//		unsigned int py = pystart;
	//		for (float y = ystart; py <= pyend; y += yPixelStep, ++py)
	//		{ 
	//			unsigned int px = pxstart;
	//			for (float x = xstart; px <= pxend; x += xPixelStep, ++px)
	//			{
	//				math::float2 pixCoord(x, y);
	//				if (dot(n0,pixCoord) + c0 > 0 &&
	//				    dot(n1,pixCoord) + c1 > 0 &&
	//				    dot(n2,pixCoord) + c2 > 0)
	//				{
	//					// TODO depth + interpolation

	//					// run fragmenmt shader
	//					FragementData data;
	//					data.depth = 0;
	//					
	//					// TODO compute interpolated values
	//					typename FragmentShader::FragementIn frag = geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[id0];

	//					if (FragmentShader::process(data, frag))
	//					{
	//						//// TODO blending...
	//						//if (px >= buffer_width || py >= buffer_height)
	//						//printf("outch: %d/%d %d/%d\n", px, buffer_width, py, buffer_height);
	//						//else
	//						surf2Dwrite(make_uchar4(255 * data.color.x, 255 * data.color.y, 255 * data.color.z, 255 * data.color.w), color_buffer, 4 * px, py);
	//					}
	//				}
	//			}
	//		}
	//	}

	//}

}


#endif //INCLUDED_FREEPIPE_FRAGMENT_PROCESSING
