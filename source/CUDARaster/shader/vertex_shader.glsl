


#version 430

#include <camera>


layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;

out vec3 c;

void main()
{
	gl_Position = camera.PV * vec4(v_position, 1.0f);
	c = v_normal;
}
