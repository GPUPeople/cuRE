


#version 450

#include <camera>
#include <object>
#include <light>

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_texture;


out vec3 a_n;
out vec2 a_tex;

void main()
{
	gl_Position = object.PVM * vec4(v_position, 1.0f);
	a_n = camera.V * vec4(v_normal, 0.0f);
	a_tex = v_texture;
}
