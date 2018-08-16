/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Shaders.hpp"
#include "defines.inl"
#include "cuda/PixelPipe.inl"

using namespace FW;


__constant__ unsigned char c_constants[sizeof(Constants)];
texture<float4, 2> t_textureAtlas;


//------------------------------------------------------------------------
// Lighting.
//------------------------------------------------------------------------

__device__ Vec3f FW::evaluateLighting(Vec3f cameraPos, Vec3f cameraNormal, const Material& material, Vec3f diffuseColor)
{
    Vec3f I = normalize(cameraPos);
    Vec3f N = normalize(cameraNormal);
    F32 dotIN = dot(I, N);
    Vec3f R = I - N * (dotIN * 2.0f);

    F32 diffuseCoef = fmaxf(-dotIN, 0.0f) * 0.75f + 0.25f;
    F32 specularCoef = powf(fmaxf(-dot(I, R), 0.0f), material.glossiness);
    return diffuseCoef * diffuseColor + specularCoef * material.specularColor;
}

//------------------------------------------------------------------------
// Vertex shaders.
//------------------------------------------------------------------------


//extern "C" __global__ void FW::vertexShader_gouraud(const InputVertex* inPtr, ShadedVertex_gouraud* outPtr, int numVertices)
//{
//    // Pick a vertex.
//
//    int vidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
//    if (vidx >= numVertices)
//        return;
//
//    const InputVertex& in = inPtr[vidx];
//    ShadedVertex_gouraud& out = outPtr[vidx];
//
//    // Shade.
//
//	Constants& constants = *(Constants*)&c_constants;
//
//	out.clipPos = constants.posToClip * Vec4f(in.modelPos, 1.0f);
//
//	Vec3f cameraPos = (constants.posToCamera * Vec4f(in.modelPos, 1.0f)).getXYZ();
//	Vec3f cameraNormal = constants.normalToCamera * in.modelNormal;
//	Material& mat = *((Material*)materialbase);
//	Vec4f diffuseColor = mat.diffuseColor;
//	Vec3f color = evaluateLighting(cameraPos, cameraNormal, mat, diffuseColor.getXYZ());
//    out.color = Vec4f(color, diffuseColor.w);
//}

extern "C" __global__ void FW::vertexShader_clipSpace(const ClipSpaceVertex* inPtr, ShadedVertex_clipSpace* outPtr, int numVertices)
{
	// Pick a vertex.

	int vidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if (vidx >= numVertices)
		return;

	const ClipSpaceVertex& in = inPtr[vidx];
	ShadedVertex_clipSpace& out = outPtr[vidx];

	// Shade.

	//Constants& constants = *(Constants*)&c_constants;

	out.clipPos = in.clipSpacePos;
	out.p = in.clipSpacePos;
	//Vec3f cameraPos = (constants.posToCamera * Vec4f(in.modelPos, 1.0f)).getXYZ();
	//Vec3f cameraNormal = constants.normalToCamera * in.modelNormal;
	//Material& mat = *((Material*)materialbase);
	//Vec4f diffuseColor = mat.diffuseColor;
	//Vec3f color = evaluateLighting(cameraPos, cameraNormal, mat, diffuseColor.getXYZ());
	//out.color = Vec4f(color, diffuseColor.w);
	//out.normal = in.clipSpacePos;
	//out.light = Vec4f(constants.normalToCamera * (constants.lightPos - in.modelPos), 1);
}

extern "C" __global__ void FW::vertexShader_eyecandy(const EyecandyVertex* inPtr, ShadedVertex_eyecandy* outPtr, int numVertices)
{
	// Pick a vertex.

	int vidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if (vidx >= numVertices)
		return;

	const EyecandyVertex& in = inPtr[vidx];
	ShadedVertex_eyecandy& out = outPtr[vidx];

	// Shade.

	//Constants& constants = *(Constants*)&c_constants;

	out.clipPos = in.pos;
	out.pos = in.pos;
	out.normal = in.normal;
	out.color = in.color;
}


extern "C" __global__ void FW::vertexShader_gouraud(const InputVertex* inPtr, ShadedVertex_gouraud* outPtr, int numVertices)
{
    // Pick a vertex.

    int vidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (vidx >= numVertices)
        return;

    const InputVertex& in = inPtr[vidx];
    ShadedVertex_gouraud& out = outPtr[vidx];

    // Shade.

	Constants& constants = *(Constants*)&c_constants;

	out.clipPos = constants.posToClip * Vec4f(in.modelPos, 1.0f);

	//Vec3f cameraPos = (constants.posToCamera * Vec4f(in.modelPos, 1.0f)).getXYZ();
	//Vec3f cameraNormal = constants.normalToCamera * in.modelNormal;
	//Material& mat = *((Material*)materialbase);
	//Vec4f diffuseColor = mat.diffuseColor;
	//Vec3f color = evaluateLighting(cameraPos, cameraNormal, mat, diffuseColor.getXYZ());
	//out.color = Vec4f(color, diffuseColor.w);
	
	out.normal = Vec4f(constants.normalToCamera * in.modelNormal, 1);
	out.light = Vec4f(constants.normalToCamera * (constants.lightPos - in.modelPos), 1);
}

//------------------------------------------------------------------------

extern "C" __global__ void FW::vertexShader_texPhong(const InputVertex* inPtr, ShadedVertex_texPhong* outPtr, int numVertices)
{
    // Pick a vertex.

    int vidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (vidx >= numVertices)
        return;

    const InputVertex& in = inPtr[vidx];
    ShadedVertex_texPhong& out = outPtr[vidx];

    // Shade.

	Constants& constants = *(Constants*)&c_constants;

    out.clipPos         = constants.posToClip * Vec4f(in.modelPos, 1.0f);
    out.cameraPos       = constants.posToCamera * Vec4f(in.modelPos, 1.0f);
    out.cameraNormal    = Vec4f(constants.normalToCamera * in.modelNormal, 0.0f);
    out.texCoord        = Vec4f(in.texCoord, 0.0f, 1.0f);
}

//------------------------------------------------------------------------
// Fragment shaders.
//------------------------------------------------------------------------

typedef GouraudShader FragmentShader_gouraud;   

typedef GouraudShaderUnlit FragmentShader_gouraud_unlit;
 
//------------------------------------------------------------------------

class FragmentShader_texPhong : public FragmentShaderBase
{
public:
    __device__ __inline__ Vec4f texture2D(const TextureSpec& spec, const Vec2f& tex, const Vec2f& texDX, const Vec2f& texDY)
    {
        // Choose LOD.

        F32 dxlen = sqr(texDX.x * spec.size.x) + sqr(texDX.y * spec.size.y);
        F32 dylen = sqr(texDY.x * spec.size.x) + sqr(texDY.y * spec.size.y);
        F32 lod = fminf(fmaxf(log2f(fmaxf(dxlen, dylen)) * 0.5f, 0.0f), (F32)(FW_ARRAY_SIZE(spec.mipLevels) - 2));
        int levelIdx = (int)lod;
        Vec4f m0 = spec.mipLevels[levelIdx + 0];
        Vec4f m1 = spec.mipLevels[levelIdx + 1];

        // Perform two bilinear lookups and interpolate.

        F32 tx = tex.x - floorf(tex.x);
        F32 ty = tex.y - floorf(tex.y);
        float4 v0 = tex2D(t_textureAtlas, tx * m0.x + m0.z, ty * m0.y + m0.w);
        float4 v1 = tex2D(t_textureAtlas, tx * m1.x + m1.z, ty * m1.y + m1.w);
        return lerp(Vec4f(v0.x, v0.y, v0.z, v0.w), Vec4f(v1.x, v1.y, v1.z, v1.w), lod - (F32)levelIdx);
    }

    __device__ __inline__ void run(void)
    {
        // Interpolate attributes.

        Vec3f cameraPos = interpolateVarying(0, m_centroid).getXYZ();
        Vec3f cameraNormal = interpolateVarying(1, m_centroid).getXYZ();
        Vec2f tex, texDX, texDY;

        if ((RENDER_MODE_FLAGS & RenderModeFlag_EnableQuads) == 0)
        {
            // Sample at pixel centroid, use analytical derivatives.
            tex = interpolateVarying(2, m_centroid).getXY();
            texDX = interpolateVarying(2, m_centroidDX).getXY();
            texDY = interpolateVarying(2, m_centroidDY).getXY();
        }
        else
        {
            // Sample at pixel center, use numerical derivatices.
            tex = interpolateVarying(2, m_center).getXY();
            texDX = dFdx(tex);
            texDY = dFdy(tex);
        }

        // Fetch material and perform texture lookups.

		Constants& constants = *(Constants*)&c_constants;

		int materialIdx = ((const S32*)constants.triangleMaterialIdx)[m_triIdx];
		const Material& material = ((const Material*)constants.materials)[materialIdx];
        Vec4f diffuseColor = material.diffuseColor;

        if (material.diffuseTexture.size.x != 0.0f)
            diffuseColor = Vec4f(texture2D(material.diffuseTexture, tex, texDX, texDY).getXYZ(), diffuseColor.w);

        if (material.alphaTexture.size.x != 0.0f)
            diffuseColor.w = texture2D(material.alphaTexture, tex, texDX, texDY).y;

        // Alpha test.
  
        if (diffuseColor.w < 0.5f)
        {
            m_discard = true;
            return;
        }

        // Shading.

        Vec3f color = evaluateLighting(cameraPos, cameraNormal, material, diffuseColor.getXYZ());
        m_color = toABGR(Vec4f(color, diffuseColor.w));
    }
};

class FragmentShader_clipSpace : public FragmentShaderBase
{
public:

	__device__ __inline__ void run(void)
	{
		//// Interpolate attributes.
		

		//if ((m_centroid.x + m_centroid.y > 0.95f && m_centroid.z < 0.05f)
		//	|| (m_centroid.x + m_centroid.z > 0.95f && m_centroid.y < 0.05f)
		//	|| (m_centroid.y + m_centroid.z > 0.95f && m_centroid.x < 0.05f))
		//{
		//	Vec3f col(1, 1, 1);
		//	m_color = toABGR(Vec4f(col, 1));
		//}
		//else
		//{
		//	m_discard = true;
		//	return;
		//}
        
		Vec4f p = interpolateVarying(0, m_centroid);
		float c = p.w * 0.01f;
		m_color = toABGR(Vec4f(c,c,c,1.0f));
	} 
};

class FragmentShader_eyecandy : public FragmentShaderBase
{
public:

	__device__ __inline__ void run(void)
	{
		//// Interpolate attributes.

		Vec4f color_ = interpolateVarying(2, m_centroid);
		Vec3f color(color_.x, color_.y, color_.z);

		if (color.x == 0 && color.y == 0 && color.z == 0)
		{
			m_discard = true;
			return;
		}

		Vec4f normal_ = interpolateVarying(1, m_centroid);
		Vec3f normal = Vec3f(normal_.x, normal_.y, normal_.z);
		normal.normalize();

		Vec4f res = Vec4f(color*(fabsf(normal.z) + 0.2f), 1.0f);

		//float v = min(min(__f[0], __f[1]), __f[2]);
		//if (v < 0.001f)
		//{
		//	res *= 1.6f;
		//}

		m_color = toABGR(res);
	}
};

//------------------------------------------------------------------------
// Pixel pipes.
//------------------------------------------------------------------------

CR_DEFINE_PIXEL_PIPE(PixelPipe_eyecandy, ShadedVertex_eyecandy, FragmentShader_eyecandy, BLEND_SHADER, SAMPLES_LOG2, RENDER_MODE_FLAGS)
CR_DEFINE_PIXEL_PIPE(PixelPipe_clipSpace, ShadedVertex_clipSpace, FragmentShader_clipSpace, BLEND_SHADER, SAMPLES_LOG2, RENDER_MODE_FLAGS)
CR_DEFINE_PIXEL_PIPE(PixelPipe_gouraud,  ShadedVertex_gouraud,  FragmentShader_gouraud,  BLEND_SHADER, SAMPLES_LOG2, RENDER_MODE_FLAGS)
CR_DEFINE_PIXEL_PIPE(PixelPipe_gouraud_unlit, ShadedVertex_gouraud, FragmentShader_gouraud_unlit, BLEND_SHADER, SAMPLES_LOG2, RENDER_MODE_FLAGS)
CR_DEFINE_PIXEL_PIPE(PixelPipe_texPhong, ShadedVertex_texPhong, FragmentShader_texPhong, BLEND_SHADER, SAMPLES_LOG2, RENDER_MODE_FLAGS)

//------------------------------------------------------------------------
