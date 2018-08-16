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

#include "App.hpp"

#include <stdio.h>
#include <conio.h>
#include <CUDA/error.h>
#include "base/helpling.h"

using namespace FW;

App::App(CUmodule module)
:   
	m_module				(module),
	m_cudaRaster			(module),
    m_colorBuffer           (NULL),
    m_depthBuffer           (NULL),
	m_drawBegin(CU::createEvent()),
	m_drawEnd(CU::createEvent()),
	m_vertexBegin(CU::createEvent()),
	m_vertexEnd(CU::createEvent())
{
    firstTimeInit();
}

void App::setData(Buffer inputVertices, Buffer shadedVertices, Buffer materials, Buffer vertexIndices, Buffer vertexMaterialsIdx, Buffer triangleMaterialIdx, int numVertices, int numMaterials, int numTriangles)
{
	m_numVertices = numVertices;
	m_numMaterials = numMaterials;
	m_numTriangles = numTriangles;

	m_inputVertices = inputVertices;
	m_shadedVertices = shadedVertices;
	m_materials = materials;
	m_vertexIndices = vertexIndices;
	m_vertexMaterialIdx = vertexMaterialsIdx;
	m_triangleMaterialIdx = triangleMaterialIdx;
}

void App::setTargets(CUarray color, CUarray depth, int resX, int resY)
{
	m_colorBuffer = color;
	m_depthBuffer = depth;
	m_resX = resX;
	m_resY = resY;

	m_cudaRaster.setSurfaces(m_colorBuffer, FW::Vec2i(m_resX, m_resY), m_depthBuffer, FW::Vec2i(m_resX, m_resY));
}

App::~App(void)
{
}

//------------------------------------------------------------------------

void App::setGlobals()
{
	// Set globals.

	{
		MutableVar<Constants> var(m_module, "c_constants");
		Constants& c = var.get();
		c.model = m_model;
		c.lightPos = m_lightpos;
		c.posToClip = m_projection * m_posToCamera;
		c.posToCamera = m_posToCamera;
		c.normalToCamera = transpose(invert(m_posToCamera.getXYZ()));
		c.materials = m_materials.address;
		c.vertexMaterialIdx = m_vertexMaterialIdx.address;
		c.triangleMaterialIdx = m_triangleMaterialIdx.address;
	}
	// m_cudaModule->setTexRef("t_textureAtlas", m_textureAtlas.getAtlasTexture().getCudaArray());
	// TODO: Add texture support?
}

double App::render(int from, int num)
{
	// Run vertex shader.
	succeed(cuEventRecord(m_drawBegin, NULL));

	FW::Array<FW::U8> param_array;
	Param a(m_inputVertices.address), b(m_shadedVertices.address), c(m_numVertices);
	const Param* params[] = { &a, &b, &c };
	setParams(params, 3, param_array);
	succeed(cuParamSetSize(m_vertexShaderKernel, param_array.getSize()));
	succeed(cuParamSetv(m_vertexShaderKernel, 0, param_array.getPtr(), param_array.getSize()));

	succeed(cuEventRecord(m_vertexBegin, NULL));
	launchKernel(m_vertexShaderKernel, m_numVertices, FW::Vec2i(0,0));
	succeed(cuEventRecord(m_vertexEnd, NULL));
	
	// Run pixel pipe.

	m_cudaRaster.setVertexBuffer(&m_shadedVertices, 0);
	Buffer kbuff = m_vertexIndices;
	kbuff.address += from * sizeof(uint32_t);
	kbuff.size = m_vertexIndices.size - from * sizeof(uint32_t);
	m_cudaRaster.setIndexBuffer(&kbuff, 0, num);
	m_cudaRaster.drawTriangles();

	succeed(cuEventRecord(m_drawEnd, NULL));
	succeed(cuEventSynchronize(m_drawEnd));


	float vert_time;
	succeed(cuEventElapsedTime(&vert_time, m_vertexBegin, m_vertexEnd));
	vert_time *= 0.001f;

	float draw_time;
	succeed(cuEventElapsedTime(&draw_time, m_drawBegin, m_drawEnd));
	draw_time *= 0.001f;

	auto stats = m_cudaRaster.getStats();

	//m_colorBuffer->resolveToScreen(gl);
	//TODO: write fill code
	return vert_time + stats.setupTime + stats.binTime + stats.coarseTime + stats.fineTime;
	//return draw_time;
}

//------------------------------------------------------------------------

void App::setPipe(const char* postfix)
{
	succeed(cuModuleGetFunction(&m_vertexShaderKernel, m_module, (String("vertexShader_") + postfix).getPtr()));
	m_cudaRaster.setPixelPipe(String("PixelPipe_") + postfix);
}

//------------------------------------------------------------------------

void App::firstTimeInit(void)
{
    //int numMSAA = 1, numModes = 2, numBlends = 2; // first 3 toggles

    //int progress = 0;
    //for (int msaa = 0; msaa < numMSAA; msaa++)
    //for (int mode = 0; mode < numModes; mode++)
    //for (int blend = 0; blend < numBlends; blend++)
    //{
    //    printf("\rPopulating CudaCompiler cache... %d/%d", ++progress, numMSAA * numModes * numBlends);
    //    m_cudaCompiler.clearDefines();
    //    m_cudaCompiler.define("SAMPLES_LOG2", msaa);
    //    m_cudaCompiler.define("RENDER_MODE_FLAGS", mode ^ RenderModeFlag_EnableDepth ^ RenderModeFlag_EnableLerp);
    //    m_cudaCompiler.define("BLEND_SHADER", (blend == 0) ? "BlendReplace" : "BlendSrcOver");
    //    m_cudaCompiler.compile(false);
    //}
    //printf("\rPopulating CudaCompiler cache... Done.\n");
    // Print footer.
    //printf("Done.\n");
    //printf("\n");
}

//------------------------------------------------------------------------
