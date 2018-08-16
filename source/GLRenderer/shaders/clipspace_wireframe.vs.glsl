


#version 450

layout(location = 0) in vec4 v_position;

out vec4 c_pos;

void main()
{
	c_pos = v_position;
}
