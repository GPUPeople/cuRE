


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../Geometry.h"

#include "EyeCandyMaterial.h"


namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace CUDARaster
{
	EyeCandyMaterial::EyeCandyMaterial(CUmodule module)
		: module(module)
	{
		matbase.diffuseColor = FW::Vec4f(1, 0, 0, 1);
		matbase.specularColor = FW::Vec3f(1, 1, 1);
		matbase.glossiness = 0.5;
		size_t size;
		succeed(cuModuleGetGlobal(&matData, &size, module, "materialbase"));
	}

	void EyeCandyMaterial::draw(const ::Geometry* geometry) const
	{
		EyeCandyGeometry* geom = (EyeCandyGeometry*)geometry;
		geom->getApp()->setPipe("eyecandy");
		cuMemcpyHtoD(matData, &matbase, sizeof(FW::Material));
		geometry->draw();
	}

	void EyeCandyMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		EyeCandyGeometry* geom = (EyeCandyGeometry*)geometry;
		geom->getApp()->setPipe("eyecandy");
		cuMemcpyHtoD(matData, &matbase, sizeof(FW::Material));
		geometry->draw(start, num_indices);
	}

	void EyeCandyMaterial::setModel(const math::float3x4& mat)
	{
	}

	void EyeCandyMaterial::setCamera(const math::float4x4& PV, const math::float3& pos)
	{
	}
}
