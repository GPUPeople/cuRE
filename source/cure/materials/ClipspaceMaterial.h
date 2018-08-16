


#ifndef INCLUDED_CURE_MATERIAL_CLIPSPACE
#define INCLUDED_CURE_MATERIAL_CLIPSPACE

#include <CUDA/module.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class ClipspaceMaterial : public Material
	{
		ClipspaceMaterial(const ClipspaceMaterial&) = delete;
		ClipspaceMaterial& operator =(const ClipspaceMaterial&) = delete;

		Pipeline& pipeline;

	public:
		ClipspaceMaterial(Pipeline& pipeline);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class VertexHeavyClipspaceMaterial : public Material
	{
		VertexHeavyClipspaceMaterial(const VertexHeavyClipspaceMaterial&) = delete;
		VertexHeavyClipspaceMaterial& operator =(const VertexHeavyClipspaceMaterial&) = delete;

		Pipeline& pipeline;
		int iterations;

	public:
		VertexHeavyClipspaceMaterial(Pipeline& pipeline, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class FragmentHeavyClipspaceMaterial : public Material
	{
		FragmentHeavyClipspaceMaterial(const FragmentHeavyClipspaceMaterial&) = delete;
		FragmentHeavyClipspaceMaterial& operator =(const FragmentHeavyClipspaceMaterial&) = delete;

		Pipeline& pipeline;
		int iterations;

	public:
		FragmentHeavyClipspaceMaterial(Pipeline& pipeline, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_CURE_MATERIAL_CLIPSPACE
