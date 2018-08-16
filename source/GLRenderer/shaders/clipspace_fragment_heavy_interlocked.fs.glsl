


#version 450

#extension GL_NV_fragment_shader_interlock : enable

#include <noise>


layout(rgba8, binding = 0) uniform image2D color_buffer;

layout(early_fragment_tests, pixel_interlock_ordered) in;

in vec4 p;

void main()
{
	float noise = simplexNoiseFractal(p.xyw);
	vec4 color = vec4(noise, noise, noise, 1.0f);

	beginInvocationInterlockNV();
	imageStore(color_buffer, ivec2(gl_FragCoord.xy), color);
	endInvocationInterlockNV();
}
