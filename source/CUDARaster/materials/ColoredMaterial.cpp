


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../Geometry.h"

#include "ColoredMaterial.h"


namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace CUDARaster
{
	ColoredMaterial::ColoredMaterial(const math::float4& color, CUmodule module) : module(module)
	{
		matbase.diffuseColor = FW::Vec4f(color.x, color.y, color.z, color.w);
		matbase.specularColor = FW::Vec3f(1,1,1);
		matbase.glossiness = 0.5;
		size_t size;
		succeed(cuModuleGetGlobal(&matData, &size, module, "materialbase"));
	}

	void ColoredMaterial::draw(const ::Geometry* geometry) const
	{
		IndexedGeometry* geom = (IndexedGeometry*)geometry;
		geom->getApp()->setPipe("gouraud_unlit");
		cuMemcpyHtoD(matData, &matbase, sizeof(FW::Material));
		geometry->draw();
	}

	void ColoredMaterial::draw(const::Geometry * geometry, int start, int num_indices) const
	{
		IndexedGeometry* geom = (IndexedGeometry*)geometry;
		geom->getApp()->setPipe("gouraud_unlit");
		cuMemcpyHtoD(matData, &matbase, sizeof(FW::Material));
		geometry->draw(start, num_indices);
	}

	void ColoredMaterial::setModel(const math::float3x4& mat)
	{
	}

	void ColoredMaterial::setCamera(const math::float4x4& PV, const math::float3& pos)
	{
	}
}
