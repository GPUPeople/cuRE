


#ifndef INCLUDED_FREEPIPE_VERTEX_SHADER
#define INCLUDED_FREEPIPE_VERTEX_SHADER

#include <cstdint>

#include <CUDA/module.h>


namespace FreePipe
{
	class Renderer;

	class VertexShader
	{
	protected:
		VertexShader(const VertexShader&) = delete;
		VertexShader& operator =(const VertexShader&) = delete;

		CUfunction kernel_geometry_processing;
		CUfunction kernel_geometry_processing_tex;
		CUfunction kernel_geometry_processing_light;
		CUfunction kernel_geometry_processing_lighttex;
		CUfunction kernel_geometry_processing_clipspace;
		
		Renderer& renderer;

	public:
		VertexShader(Renderer& renderer, CUmodule module);

		void run(unsigned int num_patches, unsigned int num_vertices, unsigned int num_indices, bool light, bool tex);

	};
}

#endif  // INCLUDED_FREEPIPE_VERTEX_SHADER
