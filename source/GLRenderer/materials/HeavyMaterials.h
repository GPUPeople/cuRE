


#ifndef INCLUDED_GLRENDERER_MATERIALS_HEAVY
#define INCLUDED_GLRENDERER_MATERIALS_HEAVY

#include <GL/buffer.h>

#include <Resource.h>


namespace GLRenderer
{
	class HeavyVertexShader;
	class HeavyFragmentShader;

	class VertexHeavyMaterial : public ::Material
	{
		VertexHeavyMaterial(const VertexHeavyMaterial&) = delete;
		VertexHeavyMaterial& operator =(const VertexHeavyMaterial&) = delete;

		const HeavyVertexShader& shader;
		GL::Buffer noise_params;

	public:
		VertexHeavyMaterial(const HeavyVertexShader& shader, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class FragmentHeavyMaterial : public ::Material
	{
		FragmentHeavyMaterial(const FragmentHeavyMaterial&) = delete;
		FragmentHeavyMaterial& operator =(const FragmentHeavyMaterial&) = delete;

		const HeavyFragmentShader& shader;
		GL::Buffer noise_params;

	public:
		FragmentHeavyMaterial(const HeavyFragmentShader& shader, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_GLRENDERER_MATERIALS_HEAVY
