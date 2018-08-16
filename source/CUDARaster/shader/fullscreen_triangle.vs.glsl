


#version 330

void main()
{
	vec2 p = vec2((gl_VertexID & 0x1), (gl_VertexID & 0x2) * 0.5f);
	gl_Position = vec4(p, 0.0f, 1.0f);
}
