


#ifndef INCLUDED_CAMERA
#define INCLUDED_CAMERA

struct Camera
{
	mat4x3 V;
	mat4x3 V_inv;
	mat4x4 P;
	mat4x4 P_inv;
	mat4x4 PV;
	mat4x4 PV_inv;
	vec3 position;
};

layout(std140, row_major, binding = 0) uniform CameraUniformBuffer
{
	Camera camera;
};

#endif  // INCLUDED_CAMERA
