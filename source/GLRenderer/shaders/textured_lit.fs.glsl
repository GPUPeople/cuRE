


#version 450


layout(location = 0) uniform vec4 albedo;
layout(binding = 0) uniform sampler2D texSampler;


in vec3 a_l;
in vec3 a_n;
in vec2 a_tex;

layout(location = 0) out vec4 color;

void main()
{
	vec3 l = normalize(a_l);
	vec3 n = normalize(a_n);

	//float lambert = max(dot(n, l), 0.0f);

	vec3 tex = texture(texSampler, a_tex).rgb;
	//color = vec4(albedo.rgb * texColor.rgb * lambert, albedo.a * texColor.a);
	//color = vec4(1, 1, 1, 1);
	color = vec4(tex * -n.z, 1.0f);
}
