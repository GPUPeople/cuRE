


#ifndef INCLUDED_GLRENDERER_MATERIAL_CLIPSPACE
#define INCLUDED_GLRENDERER_MATERIAL_CLIPSPACE

#include <GL/buffer.h>

#include <Resource.h>


namespace GLRenderer
{
	class ClipspaceShader;
	class VertexHeavyClipspaceShader;
	class FragmentHeavyClipspaceShader;


	class ClipspaceMaterial : public ::Material
	{
		ClipspaceMaterial(const ClipspaceMaterial&) = delete;
		ClipspaceMaterial& operator =(const ClipspaceMaterial&) = delete;

		const ClipspaceShader& shader;

	public:
		ClipspaceMaterial(const ClipspaceShader& shader);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class VertexHeavyClipspaceMaterial : public ::Material
	{
		VertexHeavyClipspaceMaterial(const VertexHeavyClipspaceMaterial&) = delete;
		VertexHeavyClipspaceMaterial& operator =(const VertexHeavyClipspaceMaterial&) = delete;

		const VertexHeavyClipspaceShader& shader;
		GL::Buffer noise_params;

	public:
		VertexHeavyClipspaceMaterial(const VertexHeavyClipspaceShader& shader, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};

	class FragmentHeavyClipspaceMaterial : public ::Material
	{
		FragmentHeavyClipspaceMaterial(const FragmentHeavyClipspaceMaterial&) = delete;
		FragmentHeavyClipspaceMaterial& operator =(const FragmentHeavyClipspaceMaterial&) = delete;

		const FragmentHeavyClipspaceShader& shader;
		GL::Buffer noise_params;

	public:
		FragmentHeavyClipspaceMaterial(const FragmentHeavyClipspaceShader& shader, int iterations);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_GLRENDERER_MATERIAL_CLIPSPACE
