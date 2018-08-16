


#ifndef INCLUDED_CURE_SHADER_STATE
#define INCLUDED_CURE_SHADER_STATE

#pragma once

#include <Camera.h>

#include <CUDA/module.h>


namespace cuRE
{
	class PipelineModule;

	class ShaderState
	{
	private:
        CUdeviceptr camera;
	public:
		ShaderState(const ShaderState&) = delete;
		ShaderState& operator =(const ShaderState&) = delete;

		ShaderState(const PipelineModule& module);

        void setCamera(const Camera::UniformBuffer& buffer);
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_STATE
