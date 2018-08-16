


#include <CUDA/error.h>

#include "PipelineModule.h"
#include "ShaderState.h"


namespace cuRE
{
	ShaderState::ShaderState(const PipelineModule& module)
		: camera(module.getGlobal("camera"))
	{
	}

    void ShaderState::setCamera(const Camera::UniformBuffer& buffer)
    {
        succeed(cuMemcpyHtoD(this->camera, &buffer, sizeof(Camera::UniformBuffer)));
    }
}
