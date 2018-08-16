


#version 450

#include <camera>
#include <object>

layout(location = 0) in vec3 v_position;


void main()
{
	gl_Position = object.PVM * vec4(v_position, 1.0f);
}
