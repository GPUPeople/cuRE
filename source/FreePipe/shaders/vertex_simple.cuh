


#ifndef INCLUDED_FREEPIPE_SHADERS_VERTEX_SIMPLE
#define INCLUDED_FREEPIPE_SHADERS_VERTEX_SIMPLE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "simple_vertexdata.cuh"


extern "C"
{
	__constant__ math::float4x4 c_VertexTransformMatrix;
	__constant__ math::float4x4 c_ModelMatrix;
	__constant__ math::float4x4 c_NormalTransformMatrix;
	__constant__ math::float3 c_LightPos;
}

namespace FreePipe
{
	namespace Shaders
	{

		class SimpleVertexShader
		{
		public:
			typedef SimpleVertexData VertexOut;

			__device__ static VertexOut process(math::float3 pos, math::float3 normal, math::float2 tex)
			{
				// simply apply matrices and pass out
				VertexOut out;
				out.pos = c_VertexTransformMatrix* math::float4(pos, 1);
				//out.texCoord = tex;
				return out;
			}
		};

		class SimpleVertexShaderLight
		{
		public:
			typedef SimpleVertexDataNormalLight VertexOut;

			__device__ static VertexOut process(math::float3 pos, math::float3 normal, math::float2 tex)
			{
				// simply apply matrices and pass out
				VertexOut out;
				out.pos = c_VertexTransformMatrix* math::float4(pos, 1);
				math::float3 outnormal = (c_NormalTransformMatrix * math::float4(normal, 0)).xyz();
				out.attributes[0] = outnormal.x;
				out.attributes[1] = outnormal.y;
				out.attributes[2] = outnormal.z;

				//math::float3 light = c_LightPos - (c_ModelMatrix * math::float4(pos, 1)).xyz();
				//out.attributes[3] = light.x;
				//out.attributes[4] = light.y;
				//out.attributes[5] = light.z;
				//out.texCoord = tex;
				return out;
			}
		};

		class SimpleVertexShaderTex
		{
		public:
			typedef SimpleVertexDataTex VertexOut;

			__device__ static VertexOut process(math::float3 pos, math::float3 normal, math::float2 tex)
			{
				// simply apply matrices and pass out
				VertexOut out;
				out.pos = c_VertexTransformMatrix* math::float4(pos, 1);
				out.attributes[0] = tex.x;
				out.attributes[1] = tex.y;
				return out;
			}
		};

		class SimpleVertexShaderLightTex
		{
		public:
			typedef SimpleVertexDataNormalLightTex VertexOut;

			__device__ static VertexOut process(math::float3 pos, math::float3 normal, math::float2 tex)
			{
				// simply apply matrices and pass out
				VertexOut out;
				out.pos = c_VertexTransformMatrix* math::float4(pos, 1);
				math::float3 outnormal = (c_NormalTransformMatrix * math::float4(normal, 0)).xyz();
				out.attributes[0] = outnormal.x;
				out.attributes[1] = outnormal.y;
				out.attributes[2] = outnormal.z;

				math::float3 light = c_LightPos - (c_ModelMatrix * math::float4(pos, 1)).xyz();
				out.attributes[3] = light.x;
				out.attributes[4] = light.y;
				out.attributes[5] = light.z;
				//out.texCoord = tex;

				out.attributes[6] = tex.x;
				out.attributes[7] = tex.y;
				return out;
			}
		};
	}
}


#endif // INCLUDED_FREEPIPE_SHADERS_VERTEX_SIMPLE
