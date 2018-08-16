


#ifndef INCLUDED_FREEPIPE_SHADERS_FRAGMENT_SIMPLE
#define INCLUDED_FREEPIPE_SHADERS_FRAGMENT_SIMPLE

#pragma once

#include "../fragment_data.cuh"
#include "simple_vertexdata.cuh"



extern "C"
{
	__constant__ math::float3 c_SimpleColorData;
}

namespace FreePipe
{
	namespace Shaders
	{

		class FragmentSimpleShader
		{
		public:

			typedef SimpleVertexData FragementIn;
			bool AllowEarlyZ = true;

			__device__ static bool process(FragementData& data, FragementIn interpolatedData)
			{
				data.color = math::float4(c_SimpleColorData, 1.0f);

				//data.color = math::float4(0.5f + 0.5f*interpolatedData.normal, 1.0f);
				return true;
			}
		};

		class FragmentTexShader
		{
		public:

			typedef SimpleVertexDataTex FragementIn;
			bool AllowEarlyZ = true;

			__device__ static bool process(FragementData& data, FragementIn interpolatedData)
			{
				float4 texColor = tex2D(t_phongTex, interpolatedData.attributes[0], interpolatedData.attributes[1]);
				data.color = math::float4(texColor.x, texColor.y, texColor.z, texColor.w) * math::float4(c_SimpleColorData, 1.0f);
				return true;
			}
		};
	}
}

#endif // INCLUDED_FREEPIPE_SHADERS_FRAGMENT_PHONG
