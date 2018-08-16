


#version 450

#extension GL_NV_fragment_shader_interlock : enable


layout(rgba8, binding = 0) uniform image2D color_buffer;

layout(early_fragment_tests, pixel_interlock_ordered) in;

in vec4 p;

void main()
{
	float c = p.w * 0.01f;

	beginInvocationInterlockNV();
	imageStore(color_buffer, ivec2(gl_FragCoord.xy), vec4(c, c, c, 1.0f));
	endInvocationInterlockNV();
}
