


#include <GL/error.h>

#include "ShaderState.h"


namespace GLRenderer
{
	ShaderState::ShaderState()
	{
		glBindBuffer(GL_UNIFORM_BUFFER, camera_uniform_buffer);
		glBufferStorage(GL_UNIFORM_BUFFER, sizeof(Camera::UniformBuffer), nullptr, GL_DYNAMIC_STORAGE_BIT);

		glBindBuffer(GL_UNIFORM_BUFFER, object_uniform_buffer);
		glBufferStorage(GL_UNIFORM_BUFFER, sizeof(ObjectUniformBuffer), nullptr, GL_DYNAMIC_STORAGE_BIT);

		glBindBuffer(GL_UNIFORM_BUFFER, light_uniform_buffer);
		glBufferStorage(GL_UNIFORM_BUFFER, sizeof(LightUniformBuffer), nullptr, GL_DYNAMIC_STORAGE_BIT);

		GL::throw_error();
	}

	void ShaderState::setCamera(const Camera::UniformBuffer& params)
	{
		camera_params = params;

		updateCameraUniformBuffer();
		updateObjectUniformBuffer();
	}

	void ShaderState::updateCameraUniformBuffer()
	{
		glBindBuffer(GL_UNIFORM_BUFFER, camera_uniform_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Camera::UniformBuffer), &camera_params);

		GL::throw_error();
	}

	void ShaderState::setObjectTransform(const math::affine_float4x4& M)
	{
		object_params.M = M;
		object_params.M_inv = inverse(M);

		updateObjectUniformBuffer();
		updateLightUniformBuffer();
	}

	void ShaderState::updateObjectUniformBuffer()
	{
		object_params.VM = camera_params.V * object_params.M;
		object_params.VM_inv = object_params.M_inv * camera_params.V_inv;
		object_params.PVM = camera_params.PV * object_params.M;
		object_params.PVM_inv = object_params.M_inv * camera_params.PV_inv;

		glBindBuffer(GL_UNIFORM_BUFFER, object_uniform_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(ObjectUniformBuffer), &object_params);

		GL::throw_error();
	}

	void ShaderState::setLight(const math::float3& pos, const math::float3& color)
	{
		light_pos = pos;
		light_params.color = math::float4(color, 1.0f);

		updateLightUniformBuffer();
	}

	void ShaderState::updateLightUniformBuffer()
	{
		light_params.position = object_params.M_inv * math::float4(light_pos, 1.0f);

		glBindBuffer(GL_UNIFORM_BUFFER, light_uniform_buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(LightUniformBuffer), &light_params);

		GL::throw_error();
	}

	void ShaderState::bind() const
	{
		glBindBufferBase(GL_UNIFORM_BUFFER, 0U, camera_uniform_buffer);
		glBindBufferBase(GL_UNIFORM_BUFFER, 1U, object_uniform_buffer);
		glBindBufferBase(GL_UNIFORM_BUFFER, 2U, light_uniform_buffer);
	}
}
