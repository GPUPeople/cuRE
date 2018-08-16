


#version 450

layout(location = 0) uniform sampler2D color_buffer;

in vec2 t;

layout(location = 0) out vec4 color;

void main()
{
	color = texture(color_buffer, t);
	//color = vec4(t, 0.0f, 1.0f);
}
