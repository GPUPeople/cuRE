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

#pragma once 
#include <cuda.h>
#include "cuda/PixelPipe.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Texture specification.
//------------------------------------------------------------------------

struct TextureSpec
{
    Vec2f       size;
    U32         pad[2];
    Vec4f       mipLevels[16];       // scaleX, scaleY, biasX, biasY
};

//------------------------------------------------------------------------
// Material parameters.
//------------------------------------------------------------------------

struct Material
{
    Vec4f       diffuseColor;
    Vec3f       specularColor;
    F32         glossiness;
    TextureSpec diffuseTexture;
    TextureSpec alphaTexture;
};

//------------------------------------------------------------------------
// Constants.
//------------------------------------------------------------------------

struct Constants
{
	Vec3f		lightPos;
	Mat4f		model;
    Mat4f       posToClip;
    Mat4f       posToCamera;
    Mat3f       normalToCamera;
    CUdeviceptr materials;              // numMaterials * Material
    CUdeviceptr vertexMaterialIdx;      // numVertices * S32
    CUdeviceptr triangleMaterialIdx;    // numTriangles * S32
};

//------------------------------------------------------------------------
// Vertex attributes.
//------------------------------------------------------------------------

struct InputVertex
{
    Vec3f       modelPos;
    Vec3f       modelNormal;
    Vec2f       texCoord;
};

//------------------------------------------------------------------------
// Clipspace vertex
//------------------------------------------------------------------------
struct ClipSpaceVertex
{
	Vec4f clipSpacePos;
};

//------------------------------------------------------------------------
// Eyecandy vertex
//------------------------------------------------------------------------
struct EyecandyVertex
{
	Vec4f pos;
	Vec4f normal;
	Vec4f color;
};

//------------------------------------------------------------------------
// Varyings.
//------------------------------------------------------------------------

//typedef GouraudVertex ShadedVertex_gouraud;

typedef GouraudLitVertex ShadedVertex_gouraud;

//------------------------------------------------------------------------

struct ShadedVertex_texPhong : ShadedVertexBase
{
    Vec4f       cameraPos;      // Varying 0.
    Vec4f       cameraNormal;   // Varying 1.
    Vec4f       texCoord;       // Varying 2.
};

//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

#if FW_CUDA

__device__              Vec3f   evaluateLighting        (Vec3f cameraPos, Vec3f cameraNormal, const Material& material, Vec3f diffuseColor);
extern "C" __global__   void    vertexShader_eyecandy   (const EyecandyVertex* inPtr, ShadedVertex_eyecandy*  outPtr, int numVertices);
extern "C" __global__   void    vertexShader_clipSpace  (const ClipSpaceVertex* inPtr, ShadedVertex_clipSpace*  outPtr, int numVertices);
extern "C" __global__   void    vertexShader_gouraud    (const InputVertex* inPtr, ShadedVertex_gouraud*  outPtr, int numVertices);
extern "C" __global__   void    vertexShader_texPhong   (const InputVertex* inPtr, ShadedVertex_texPhong* outPtr, int numVertices);

// CR_DEFINE_PIXEL_PIPE(PixelPipe_gouraud, ...)
// CR_DEFINE_PIXEL_PIPE(PixelPipe_texPhong, ...)

#endif

//------------------------------------------------------------------------
}
