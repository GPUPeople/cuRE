


#version 330

uniform samplerBuffer tex_color;
uniform ivec2 render_target_size;


layout(location = 0) out vec4 color;

void main()
{
	color = texelFetch(tex_color, int(gl_FragCoord.y) * render_target_size.x + int(gl_FragCoord.x));
}
