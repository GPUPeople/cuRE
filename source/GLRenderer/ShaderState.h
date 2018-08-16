


#ifndef INCLUDED_GLRENDERER_SHADER_STATE
#define INCLUDED_GLRENDERER_SHADER_STATE

#pragma once

#include <GL/buffer.h>

#include <Camera.h>


namespace GLRenderer
{
	class ShaderState
	{
	private:
		struct ObjectUniformBuffer
		{
			__declspec(align(64)) math::affine_float4x4 M;
			__declspec(align(64)) math::affine_float4x4 M_inv;
			__declspec(align(64)) math::affine_float4x4 VM;
			__declspec(align(64)) math::affine_float4x4 VM_inv;
			__declspec(align(64)) math::float4x4 PVM;
			__declspec(align(64)) math::float4x4 PVM_inv;
		};

		struct LightUniformBuffer
		{
			math::float4 position;
			math::float4 color;
		};

		GL::Buffer camera_uniform_buffer;
		GL::Buffer object_uniform_buffer;
		GL::Buffer light_uniform_buffer;

		Camera::UniformBuffer camera_params;
		ObjectUniformBuffer object_params;
		LightUniformBuffer light_params;

		math::float3 light_pos;

		void updateCameraUniformBuffer();
		void updateObjectUniformBuffer();
		void updateLightUniformBuffer();

	public:
		ShaderState(const ShaderState&) = delete;
		ShaderState& operator =(const ShaderState&) = delete;

		ShaderState();

		void setCamera(const Camera::UniformBuffer& params);
		void setObjectTransform(const math::affine_float4x4& M);
		void setLight(const math::float3& pos, const math::float3& color);

		void bind() const;
	};
}

#endif  // INCLUDED_GLRENDERER_SHADER_STATE
