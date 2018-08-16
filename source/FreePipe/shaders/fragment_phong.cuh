


#ifndef INCLUDED_FREEPIPE_SHADERS_FRAGMENT_PHONG
#define INCLUDED_FREEPIPE_SHADERS_FRAGMENT_PHONG

#pragma once

#include "../fragment_data.cuh"
#include "simple_vertexdata.cuh"
#include "fragment_phong.h"


extern "C"
{
	__constant__ FreePipe::Shaders::PhongData c_PhongData;
	texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> t_phongTex;
}

namespace FreePipe
{
	namespace Shaders
	{

		class FragmentPhongShader
		{
		public:

			typedef SimpleVertexDataNormalLight FragementIn;
			bool AllowEarlyZ = true;

			__device__ static bool process(FragementData& data, FragementIn interpolatedData)
			{
				math::float3 n(interpolatedData.attributes[0], interpolatedData.attributes[1], interpolatedData.attributes[2]);
				//n = normalize(n);
				//math::float3 l(interpolatedData.attributes[3], interpolatedData.attributes[4], interpolatedData.attributes[5]);
				//l = normalize(l);

				//float lambert = max(dot(n, l), 0.0f);


				//data.color = math::float4(c_PhongData.materialDiffuseColor * lambert, 1.0f);

				data.color = math::float4(0.5f + 0.5f*n, 1.0f);
				return true;
			}
		};

		class FragmentPhongTexShader
		{
		public:

			typedef SimpleVertexDataNormalLightTex FragementIn;
			bool AllowEarlyZ = true;

			__device__ static bool process(FragementData& data, FragementIn interpolatedData)
			{
				math::float3 n(interpolatedData.attributes[0], interpolatedData.attributes[1], interpolatedData.attributes[2]);
				n = normalize(n);
				math::float3 l(interpolatedData.attributes[3], interpolatedData.attributes[4], interpolatedData.attributes[5]);
				l = normalize(l);

				float diffuse = max(dot(n, l), 0.0f);
				float4 texColor = tex2D(t_phongTex, interpolatedData.attributes[6], interpolatedData.attributes[7]);

				data.color = math::float4(texColor.x, texColor.y, texColor.z, texColor.w) * math::float4(c_PhongData.materialDiffuseColor * diffuse, 1.0f);

				//data.color = math::float4(0.5f + 0.5f*interpolatedData.normal, 1.0f);
				return true;
			}
		};
	}
}

#endif // INCLUDED_FREEPIPE_SHADERS_FRAGMENT_PHONG
