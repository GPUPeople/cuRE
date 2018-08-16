


#ifndef INCLUDED_FREEPIPE_MATERIAL_CLIPSPACE
#define INCLUDED_FREEPIPE_MATERIAL_CLIPSPACE

#include <CUDA/module.h>

#include <Resource.h>


namespace FreePipe
{
	class Renderer;

	class ClipspaceMaterial : public Material
	{
		ClipspaceMaterial(const ClipspaceMaterial&) = delete;
		ClipspaceMaterial& operator=(const ClipspaceMaterial&) = delete;

		CUmodule module;

		CUfunction kernel_fragment_processing;

		Renderer& renderer;

	public:
		ClipspaceMaterial(Renderer& renderer, CUmodule module);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif // INCLUDED_FREEPIPE_MATERIAL_CLIPSPACE
