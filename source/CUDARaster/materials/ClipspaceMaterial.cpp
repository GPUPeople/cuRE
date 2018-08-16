


#include <CUDA/module.h>
#include <CUDA/error.h>

#include "../Geometry.h"

#include "ClipspaceMaterial.h"


namespace
{
	unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1U) / b;
	}
}

namespace CUDARaster
{
	ClipspaceMaterial::ClipspaceMaterial(CUmodule module)
		: module(module)
	{
		matbase.diffuseColor = FW::Vec4f(1, 0, 0, 1);
		matbase.specularColor = FW::Vec3f(1, 1, 1);
		matbase.glossiness = 0.5;
		size_t size;
		succeed(cuModuleGetGlobal(&matData, &size, module, "materialbase"));
	}

	void ClipspaceMaterial::draw(const ::Geometry* geometry) const
	{
		ClipspaceGeometry* geom = (ClipspaceGeometry*)geometry;
		geom->getApp()->setPipe("clipSpace");
		cuMemcpyHtoD(matData, &matbase, sizeof(FW::Material));
		geometry->draw();
	}

	void ClipspaceMaterial::draw(const ::Geometry* geometry, int start, int num_indices) const
	{
		ClipspaceGeometry* geom = (ClipspaceGeometry*)geometry;
		geom->getApp()->setPipe("clipSpace");
		cuMemcpyHtoD(matData, &matbase, sizeof(FW::Material));
		geometry->draw(start, num_indices);
	}

	void ClipspaceMaterial::setModel(const math::float3x4& mat)
	{
	}

	void ClipspaceMaterial::setCamera(const math::float4x4& PV, const math::float3& pos)
	{
	}
}
