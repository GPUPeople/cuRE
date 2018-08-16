


#ifndef INCLUDED_GLRENDERER_MATERIAL_EYECANDY
#define INCLUDED_GLRENDERER_MATERIAL_EYECANDY

#include <GL/buffer.h>

#include <Resource.h>


namespace GLRenderer
{
	class EyeCandyShader;
	class VertexHeavyEyeCandyShader;
	class FragmentHeavyEyeCandyShader;

	class EyeCandyMaterial : public ::Material
	{
		EyeCandyMaterial(const EyeCandyMaterial&) = delete;
		EyeCandyMaterial& operator=(const EyeCandyMaterial&) = delete;

		const EyeCandyShader& shader;

	public:
		EyeCandyMaterial(const EyeCandyShader& shader);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class VertexHeavyEyeCandyMaterial : public ::Material
	{
		VertexHeavyEyeCandyMaterial(const VertexHeavyEyeCandyMaterial&) = delete;
		VertexHeavyEyeCandyMaterial& operator=(const VertexHeavyEyeCandyMaterial&) = delete;

		const VertexHeavyEyeCandyShader& shader;
		GL::Buffer noise_params;

	public:
		VertexHeavyEyeCandyMaterial(const VertexHeavyEyeCandyShader& shader, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class FragmentHeavyEyeCandyMaterial : public ::Material
	{
		FragmentHeavyEyeCandyMaterial(const FragmentHeavyEyeCandyMaterial&) = delete;
		FragmentHeavyEyeCandyMaterial& operator=(const FragmentHeavyEyeCandyMaterial&) = delete;

		const FragmentHeavyEyeCandyShader& shader;
		GL::Buffer noise_params;

	public:
		FragmentHeavyEyeCandyMaterial(const FragmentHeavyEyeCandyShader& shader, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif // INCLUDED_GLRENDERER_MATERIAL_CLIPSPACE
