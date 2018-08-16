


#version 450

layout(location = 0) in vec4 v_position;

out vec4 p;

void main()
{
	gl_Position = v_position;
	p = v_position;
}
