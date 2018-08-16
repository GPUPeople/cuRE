


#version 450

#include <noise>


layout(location = 0) in vec4 v_position;

out vec3 c;

void main()
{
	gl_Position = v_position;

	float noise = simplexNoiseFractal(v_position.xyw);
	c = vec3(noise, noise, noise);
}
