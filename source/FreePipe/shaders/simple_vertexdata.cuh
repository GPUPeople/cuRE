


#ifndef INCLUDED_FREEPIPE_SHADERS_SIMPLEVERTEXDATA
#define INCLUDED_FREEPIPE_SHADERS_SIMPLEVERTEXDATA

#pragma once

#include <math/vector.h>


namespace FreePipe
{
	namespace Shaders
	{
		template <unsigned int TInterpolators>
		struct SimpleVertexDataWithAttributes
		{
			static const int Interpolators = TInterpolators;
			SimpleVertexDataWithAttributes() = default;
			math::float4 pos;
			float attributes[TInterpolators];
		};
		template <>
		struct SimpleVertexDataWithAttributes<0>
		{
			static const int Interpolators = 0;
			SimpleVertexDataWithAttributes() = default;
			math::float4 pos;
		};

		typedef SimpleVertexDataWithAttributes<0> SimpleVertexData;
		typedef SimpleVertexDataWithAttributes<4> SimpleVertexData4D;
		typedef SimpleVertexDataWithAttributes<6> SimpleVertexDataNormalLight;
		typedef SimpleVertexDataWithAttributes<2> SimpleVertexDataTex;
		typedef SimpleVertexDataWithAttributes<8> SimpleVertexDataNormalLightTex;
	}
}

#endif // INCLUDED_FREEPIPE_SHADERS_VERTEX_SIMPLE
