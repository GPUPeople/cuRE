


#ifndef INCLUDED_FREEPIPE_SHADERS_FRAGMENT_CLIPSPACE
#define INCLUDED_FREEPIPE_SHADERS_FRAGMENT_CLIPSPACE

#pragma once

#include "../fragment_data.cuh"
#include "simple_vertexdata.cuh"
#include "fragment_phong.h"


namespace FreePipe
{
	namespace Shaders
	{
		class ClipSpaceVertexShader
		{
		public:
			typedef SimpleVertexData4D VertexOut;

			__device__ static VertexOut process(math::float4 pos)
			{
				// simply apply matrices and pass out
				VertexOut out;
				out.pos = pos;

				out.attributes[0] = pos.x;
				out.attributes[1] = pos.y;
				out.attributes[2] = pos.z;
				out.attributes[3] = pos.w;
				return out;
			}
		};

		class ClipSpaceFragmentShader
		{
		public:
			typedef SimpleVertexData4D FragementIn;
			bool AllowEarlyZ = false;

			__device__ static bool process(FragementData& data, FragementIn interpolatedData)
			{
				float w = interpolatedData.attributes[3] * 0.01f;
				data.color = math::float4(w, w, w, 1.0f);
				return true;
			}
		};
	}
}

#endif // INCLUDED_FREEPIPE_SHADERS_FRAGMENT_CLIPSPACE
