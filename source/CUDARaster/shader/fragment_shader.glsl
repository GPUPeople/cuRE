


#version 430


in vec3 c;

layout(location = 0) out vec4 color;

void main()
{
	color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	//color = vec4(c * 0.5f + 0.5f, 1.0f);
}
