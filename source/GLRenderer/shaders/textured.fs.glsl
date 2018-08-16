


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

	float lambert = max(dot(n, l), 0.0f);
	vec4 texColor = texture(texSampler, a_tex);
	//color = vec4(albedo.rgb * lambert, albedo.a);
	color = vec4(1, 1, 1, 1);
}
