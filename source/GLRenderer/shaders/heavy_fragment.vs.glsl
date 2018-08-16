


#version 450

#include <camera>
#include <object>

layout(location = 0) in vec3 v_position;

out vec3 p;

void main()
{
	gl_Position = object.PVM * vec4(v_position, 1.0f);
	p = v_position;
}
