


#version 450


layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec4 c_pos[3];

out vec3 pos;

void main()
{
	pos = c_pos[0].xyw;
	gl_Position = c_pos[0];
	EmitVertex();

	pos = c_pos[1].xyw;
	gl_Position = c_pos[1];
	EmitVertex();

	pos = c_pos[2].xyw;
	gl_Position = c_pos[2];
	EmitVertex();
}
