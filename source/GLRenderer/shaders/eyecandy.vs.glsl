


#version 450

layout(location = 0) in vec4 v_position;
layout(location = 1) in vec4 v_normal;
layout(location = 2) in vec4 v_color;

out vec4 v_p;
out vec3 v_n;
out vec3 v_c;

void main()
{
	gl_Position = v_position;
	//pos = vec3(v_position.xy * v_position.w, v_position.w);

	v_n = v_normal.xyz;
	v_c = v_color.xyz;

	v_p = v_position;
	//gl_Position = vec4(v_position.x, v_position.y, v_position.z+0.7f * v_position.w, v_position.w);
}
