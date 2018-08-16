


#version 450


layout(location = 0) uniform vec4 albedo;

//in vec3 a_l;
in vec3 a_n;

layout(location = 0) out vec4 color;

void main()
{
	//vec3 l = normalize(a_l);
	//vec3 n = normalize(a_n);

	//float lambert = max(dot(n, l), 0.0f);

	vec3 ndx = dFdx(a_n);
	vec3 ndy = dFdy(a_n);
	float lnx = length(ndx);
	float lny = length(ndy);

	color = vec4(abs(normalize(ndx)), 1.0f);

	color = vec4(0.5 + 0.5*a_n.x, 0.5 + 0.5*a_n.y, 0.5 + 0.5*a_n.z, 1.0);
	//color = vec4(albedo.rgb * lambert, albedo.a);
	//color = vec4(1, 1, 1, 1);

	//color = vec4(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z, 1.0f);
}
