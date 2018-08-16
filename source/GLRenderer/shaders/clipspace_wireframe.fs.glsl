


#version 450


in vec3 pos;

layout(location = 0) out vec4 color;

void main()
{
	vec2 dd = abs(vec2(dFdx(gl_FragCoord.w), dFdy(gl_FragCoord.w)));

	vec3 n = vec3(dd.x, dd.y, 0.0f);// 1.0f - length(dd));
	//vec3 n = vec3(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z);

	//color = vec4(n*1000, 1.0f);
	color = vec4(mod(pos, 20.0f) * 0.02f, 1.0f);

	//color = vec4(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z, 0.25f);
}
