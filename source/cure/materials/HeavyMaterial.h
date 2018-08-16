


#ifndef INCLUDED_CURE_MATERIAL_HEAVY
#define INCLUDED_CURE_MATERIAL_HEAVY

#include <CUDA/module.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class VertexHeavyMaterial : public Material
	{
		VertexHeavyMaterial(const VertexHeavyMaterial&) = delete;
		VertexHeavyMaterial& operator =(const VertexHeavyMaterial&) = delete;

		Pipeline& pipeline;
		int iterations;

	public:
		VertexHeavyMaterial(Pipeline& pipeline, int iterations);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class FragmentHeavyMaterial : public Material
	{
		FragmentHeavyMaterial(const FragmentHeavyMaterial&) = delete;
		FragmentHeavyMaterial& operator =(const FragmentHeavyMaterial&) = delete;

		Pipeline& pipeline;
		int iterations;

	public:
		FragmentHeavyMaterial(Pipeline& pipeline, int iterations);

		void draw(const ::Geometry* geometry) const;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_CURE_MATERIAL_HEAVY
