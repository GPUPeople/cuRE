


#ifndef INCLUDED_GLRENDERER_MATERIAL_CLIPSPACE
#define INCLUDED_GLRENDERER_MATERIAL_CLIPSPACE

#include <Resource.h>


namespace GLRenderer
{
	class ClipspaceShader;

	class ClipspaceMaterial : public ::Material
	{
	protected:
		ClipspaceMaterial(const ClipspaceMaterial&) = delete;
		ClipspaceMaterial& operator =(const ClipspaceMaterial&) = delete;

		const ClipspaceShader& shader;

	public:
		ClipspaceMaterial(const ClipspaceShader& shader);

		void draw(const ::Geometry* geometry) const;
	};
}

#endif  // INCLUDED_GLRENDERER_MATERIAL_CLIPSPACE
