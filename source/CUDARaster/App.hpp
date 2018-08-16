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

#include <CUDA/event.h>

#include "Shaders.hpp"
#include "CudaRaster.hpp"

//#define CLIPSPACE_GEOM 0
//#define EYECANDY_GEOM 1

namespace FW
{
//------------------------------------------------------------------------

class App
{
public:
						App				(CUmodule module);
						~App			(void);

	void				setData			(Buffer inputVertices, Buffer shadedVertices, Buffer materials, Buffer vertexIndices, Buffer vertexMaterialsIdx, Buffer triangleMaterialIdx, int numVertices, int numMaterials, int numTriangles);

	void				setTargets		(CUarray color, CUarray depth, int resX, int resY);

	void                setGlobals();

	double                render(int from, int num);

	void				setProjection	(Mat4f& projection) { m_projection = projection;	}

	void				immediateClearColor(Vec4f color) { m_cudaRaster.immediateClearColor(color); }

	void				immediateClearDepth(F32 depth) { m_cudaRaster.immediateClearDepth(depth); }

	void				setPosToCam     (Mat4f& pos_to_cam) { m_posToCamera = pos_to_cam;	}

	void				setClear(bool clear, const FW::Vec4f& color = (0.0F), F32 depth = (1.0F)) { m_cudaRaster.deferredClear(clear, color, depth);	}

	void				setLight		(const FW::Vec3f& pos, const FW::Vec3f& color) { m_lightpos = pos;	}

	void				setModel		(Mat4f& model) { m_model = model;	}

	void				setPipe	(const char* postfix);

private:

    void                firstTimeInit   (void);

private:
                        App             (const App&); // forbidden
    App&                operator=       (const App&); // forbidden

private:
    CudaRaster          m_cudaRaster;

    // State.

	CUmodule			m_module;

    // Mesh.

	Vec3f				m_lightpos;
    S32                 m_numVertices;
    S32                 m_numMaterials;
    S32                 m_numTriangles;
    Buffer              m_inputVertices;
    Buffer              m_shadedVertices;
    Buffer              m_materials;
    Buffer              m_vertexIndices;
    Buffer              m_vertexMaterialIdx;
    Buffer              m_triangleMaterialIdx;
    //TextureAtlas        m_textureAtlas;

    // Pipe.

	Mat4f				m_model;
	Mat4f				m_posToCamera;
	Mat4f				m_projection;

	CUfunction          m_vertexShaderKernel;

	U32					m_resX;
	U32					m_resY;

    CUarray		        m_colorBuffer;
	CUarray		        m_depthBuffer;

	CU::unique_event          m_drawBegin;
	CU::unique_event          m_drawEnd;
	CU::unique_event          m_vertexBegin;
	CU::unique_event          m_vertexEnd;
};

//------------------------------------------------------------------------
}
