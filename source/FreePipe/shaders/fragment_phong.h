


#ifndef INCLUDED_FREEPIPE_SHADERS_FRAGMENT_PHONG_HOST
#define INCLUDED_FREEPIPE_SHADERS_FRAGMENT_PHONG_HOST

#pragma once

#include <math/vector.h>
#include <math/matrix.h>



namespace FreePipe
{
	namespace Shaders
	{
		struct PhongData
		{
			math::float3 materialDiffuseColor;
			math::float3 materialSpecularColor;
			float diffuseAlpha;
			float specularAlpha;
			float specularExp;

			math::float3 lightColor;
		};
	}
}

#endif // INCLUDED_FREEPIPE_SHADERS_FRAGMENT_PHONG_HOST
