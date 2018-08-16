


#ifndef INCLUDED_CUDA_GL_GRAPHICS_RESOURCE
#define INCLUDED_CUDA_GL_GRAPHICS_RESOURCE

#pragma once

#include <GL/gl.h>

#include <cuda.h>
#include <cudaGL.h>

#include "graphics_resource.h"


namespace CU
{
	namespace graphics
	{
		unique_resource registerGLBuffer(GLuint buffer, unsigned int flags = CU_GRAPHICS_REGISTER_FLAGS_NONE);
		unique_resource registerGLImage(GLuint image, GLenum target, unsigned int flags = CU_GRAPHICS_REGISTER_FLAGS_NONE);
	}
}

#endif  // INCLUDED_CUDA_GL_GRAPHICS_RESOURCE
