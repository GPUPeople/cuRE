


#version 450

out vec2 t;

void main()
{
	vec2 p = vec2((gl_VertexID & 0x2) * 0.5f, (gl_VertexID & 0x1));
	gl_Position = vec4(p * 4.0f - 1.0f, 0.0f, 1.0f);
	t = 2.0f * p;
}
