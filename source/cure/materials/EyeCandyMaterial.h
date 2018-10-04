


#ifndef INCLUDED_CURE_MATERIAL_EYECANDY
#define INCLUDED_CURE_MATERIAL_EYECANDY

#include <CUDA/module.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class EyeCandyMaterial : public Material
	{
		EyeCandyMaterial(const EyeCandyMaterial&) = delete;
		EyeCandyMaterial& operator=(const EyeCandyMaterial&) = delete;

		Pipeline& pipeline;

	public:
		EyeCandyMaterial(Pipeline& pipeline);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class VertexHeavyEyeCandyMaterial : public Material
	{
		VertexHeavyEyeCandyMaterial(const VertexHeavyEyeCandyMaterial&) = delete;
		VertexHeavyEyeCandyMaterial& operator=(const VertexHeavyEyeCandyMaterial&) = delete;

		Pipeline& pipeline;
		int iterations;

	public:
		VertexHeavyEyeCandyMaterial(Pipeline& pipeline, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class FragmentHeavyEyeCandyMaterial : public Material
	{
		FragmentHeavyEyeCandyMaterial(const FragmentHeavyEyeCandyMaterial&) = delete;
		FragmentHeavyEyeCandyMaterial& operator=(const FragmentHeavyEyeCandyMaterial&) = delete;

		Pipeline& pipeline;
		int iterations;

	public:
		FragmentHeavyEyeCandyMaterial(Pipeline& pipeline, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif // INCLUDED_CURE_MATERIAL_CLIPSPACE
