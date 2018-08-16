


#version 450

#include <camera>
#include <object>
#include <light>

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;


out vec3 a_n;

void main()
{
	gl_Position = object.PVM * vec4(v_position, 1.0f);
	a_n = camera.V * vec4(v_normal, 0.0f);
}
