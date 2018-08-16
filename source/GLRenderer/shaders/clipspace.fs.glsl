


#version 450


in vec4 p;

layout(location = 0) out vec4 color;

void main()
{
	float c = p.w * 0.01f;
	color = vec4(c, c, c, 1.0f);
}
