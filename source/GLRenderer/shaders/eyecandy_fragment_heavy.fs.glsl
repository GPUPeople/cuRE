


#version 450

#include <noise>


in vec3 p;

layout(location = 0) out vec4 color;

void main()
{
	float noise = simplexNoiseFractal(p);
	color = vec4(noise, noise, noise, 1.0f);
}
