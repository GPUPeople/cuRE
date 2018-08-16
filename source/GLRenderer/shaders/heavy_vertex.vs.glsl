


#version 450

#include <camera>
#include <object>
#include <noise>

layout(location = 0) in vec3 v_position;

out vec3 c;

void main()
{
	gl_Position = object.PVM * vec4(v_position, 1.0f);

	float noise = simplexNoiseFractal(v_position);
	c = vec3(noise, noise, noise);
}
