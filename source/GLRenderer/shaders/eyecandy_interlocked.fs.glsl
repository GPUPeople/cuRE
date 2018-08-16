


#version 450

#extension GL_NV_fragment_shader_interlock : enable


layout(rgba8, binding = 0) uniform image2D color_buffer;

layout(early_fragment_tests, pixel_interlock_ordered) in;

in vec4 v_p;
in vec3 v_n;
in vec3 v_c;

void main()
{
	vec4 res = vec4(v_c * (-v_n.z + 0.2f), 1.0f);

	beginInvocationInterlockNV();
	imageStore(color_buffer, ivec2(gl_FragCoord.xy), res);
	endInvocationInterlockNV();
}
