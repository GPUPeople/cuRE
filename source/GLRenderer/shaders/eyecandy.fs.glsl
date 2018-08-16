


#version 450


in vec4 v_p;
in vec3 v_n;
in vec3 v_c;

layout(location = 0) out vec4 color;

void main()
{
	color = vec4(v_c * (-v_n.z + 0.2f), 1.0f);
}
