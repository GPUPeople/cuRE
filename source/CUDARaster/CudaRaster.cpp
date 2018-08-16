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

#include <stdexcept>

#include "CudaRaster.hpp"
#include "base/defs.hpp"
#include "base/helpling.h"
#include <CUDA/error.h>
#include <CUDA/launch.h>
#include <iostream>

using namespace FW;

//------------------------------------------------------------------------

static const struct
{
	const char* format;
} g_profCounters[] =
{
#define LAMBDA(ID, FORMAT) { FORMAT },
	CR_PROFILING_COUNTERS(LAMBDA)
#undef LAMBDA
};

//------------------------------------------------------------------------

static const struct
{
	S32         parent;
	const char* format;
} g_profTimers[] =
{
#define LAMBDA(ID, PARENT, FORMAT) { (int)&((CRProfTimerOrder*)NULL)->PARENT, FORMAT },
	CR_PROFILING_TIMERS(LAMBDA)
#undef LAMBDA
};

//------------------------------------------------------------------------

CudaRaster::CudaRaster(CUmodule module)
: 
m_module(module),
m_colorBuffer(NULL),
m_depthBuffer(NULL),

m_deferredClear(false),
m_clearColor(0),
m_clearDepth(0),

m_vertexBuffer(NULL),
m_vertexOfs(0),
m_indexBuffer(NULL),
m_indexOfs(0),
m_numTris(0),

m_numSMs(1),
m_numFineWarps(1),

m_maxSubtris(1),
m_maxBinSegs(1),
m_maxTileSegs(1)
{
	// Check CUDA version, compute capability, and NVCC availability.
	
	int version;
	cuDriverGetVersion(&version);

	if (version < 40)
	{	fail("CudaRaster: CUDA 4.0 or later is required!");	}

	CUdevice dev;
	cuDeviceGet(&dev, 0);
	struct { int major; int minor; } s;

	cuDeviceComputeCapability(&s.major, &s.minor, dev);

	if (s.major < 2)
	{	fail("CudaRaster: Compute capability 2.0 or better is required!");	}

	// Create CUDA events.

	succeed(cuEventCreate(&m_evSetupBegin, 0));
	succeed(cuEventCreate(&m_evBinBegin, 0));
	succeed(cuEventCreate(&m_evCoarseBegin, 0));
	succeed(cuEventCreate(&m_evFineBegin, 0));
	succeed(cuEventCreate(&m_evFineEnd, 0));

	// Allocate fixed-size buffers.

	resizeDiscard(m_binFirstSeg, CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(S32));
	resizeDiscard(m_binTotal, CR_MAXBINS_SQR * CR_BIN_STREAMS_SIZE * sizeof(S32));
	resizeDiscard(m_activeTiles, CR_MAXTILES_SQR * sizeof(S32));
	resizeDiscard(m_tileFirstSeg, CR_MAXTILES_SQR * sizeof(S32));
}

//------------------------------------------------------------------------

CudaRaster::~CudaRaster(void)
{
	succeed(cuEventDestroy(m_evSetupBegin));
	succeed(cuEventDestroy(m_evBinBegin));
	succeed(cuEventDestroy(m_evCoarseBegin));
	succeed(cuEventDestroy(m_evFineBegin));
	succeed(cuEventDestroy(m_evFineEnd));
}

//------------------------------------------------------------------------

void CudaRaster::setSurfaces(CUarray color, FW::Vec2i colorBuffer_size,
							 CUarray depth, FW::Vec2i depthBuffer_size)
{
	m_colorBuffer = color;
	m_depthBuffer = depth;
	if (!m_colorBuffer && !m_depthBuffer)
		return;

	// Check for errors.

	if (!m_colorBuffer)
	{	fail("CudaRaster: No color buffer specified!");	}

	if (!m_depthBuffer)
	{	fail("CudaRaster: No depth buffer specified!");	}

	if (colorBuffer_size != depthBuffer_size)
	{	fail("CudaRaster: Mismatch in size between surfaces!");	}

	// Initialize parameters.

	m_viewportSize = colorBuffer_size;
	m_sizePixels = (colorBuffer_size + CR_TILE_SIZE - 1) & -CR_TILE_SIZE;
	m_sizeTiles = m_sizePixels >> CR_TILE_LOG2;
	m_numTiles = m_sizeTiles.x * m_sizeTiles.y;
	m_sizeBins = (m_sizeTiles + CR_BIN_SIZE - 1) >> CR_BIN_LOG2;
	m_numBins = m_sizeBins.x * m_sizeBins.y;
}

//------------------------------------------------------------------------

void CudaRaster::deferredClear(bool deferred, const Vec4f& color, F32 depth)
{
	m_deferredClear = deferred;
	m_clearColor = color.toABGR();
	m_clearDepth = encodeDepth((U32)min((U64)(depth * exp2(32)), (U64)FW_U32_MAX));
}

int divup(int a, int b)
{
	return ((a + b - 1) / b);
}

void CudaRaster::immediateClearColor(const Vec4f& color)
{
	if (!m_colorBuffer)
		throw std::runtime_error("What, there is no color buffer to clear?");

	setSurfRef("s_colorBuffer", m_module, m_colorBuffer);
	
	CU::Function<std::uint32_t, std::uint32_t, std::uint32_t> clear_func { CU::getFunction(m_module, "clearRGBA") };

	CU::dim blocksize = { 16U, 16U };
	CU::dim gridsize = { static_cast<unsigned int>(divup(m_viewportSize.x, blocksize[0])), static_cast<unsigned int>(divup(m_viewportSize.y, blocksize[1])) };
	clear_func(gridsize, blocksize, 0U, nullptr, color.toABGR(), m_viewportSize.x, m_viewportSize.y);
}

void CudaRaster::immediateClearDepth(const F32 depth)
{
	if (!m_depthBuffer)
		throw std::runtime_error("What, there is no depth buffer to clear?");

	setSurfRef("s_depthBuffer", m_module, m_depthBuffer);

	CU::Function<std::uint32_t, std::uint32_t, std::uint32_t> clear_func{ CU::getFunction(m_module, "clearDepth") };

	U32 d = encodeDepth((U32)min((U64)(depth * exp2(32)), (U64)FW_U32_MAX));
	CU::dim blocksize = { 16U, 16U };
	CU::dim gridsize = { static_cast<unsigned int>(divup(m_viewportSize.x, blocksize[0])), static_cast<unsigned int>(divup(m_viewportSize.y, blocksize[1])) };
	clear_func(gridsize, blocksize, 0U, nullptr, d, m_viewportSize.x, m_viewportSize.y);
}

//------------------------------------------------------------------------

void CudaRaster::setPixelPipe(const String& name)
{
	// Query kernels.

	succeed(cuModuleGetFunction(&m_setupKernel, m_module, (name + "_triangleSetup").getPtr()));
	succeed(cuModuleGetFunction(&m_binKernel, m_module, (name + "_binRaster").getPtr()));
	succeed(cuModuleGetFunction(&m_coarseKernel, m_module, (name + "_coarseRaster").getPtr()));
	succeed(cuModuleGetFunction(&m_fineKernel, m_module, (name + "_fineRaster").getPtr()));

	// Query spec.

	getGlobal<PixelPipeSpec>(m_module, (name + "_spec").getPtr(), m_pipeSpec);

	// Query launch bounds.

	CUdevice dev;
	cuDeviceGet(&dev, 0);
	succeed(cuDeviceGetAttribute(&m_numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
	succeed(cuFuncGetAttribute(&m_numFineWarps, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_fineKernel));
	m_numFineWarps = min(m_numFineWarps / 32, CR_FINE_MAX_WARPS);
}

//------------------------------------------------------------------------

void CudaRaster::setVertexBuffer(Buffer* buf, S64 ofs)
{
	m_vertexBuffer = buf;
	m_vertexOfs = ofs;
}

//------------------------------------------------------------------------

void CudaRaster::setIndexBuffer(Buffer* buf, S64 ofs, int numTris)
{
	m_indexBuffer = buf;
	m_indexOfs = ofs;
	m_numTris = numTris;
}

//------------------------------------------------------------------------

void CudaRaster::drawTriangles(void)
{
	m_deferredClear = false;

	int maxSubtrisSlack = 4096;     // x 81B    = 324KB
	int maxBinSegsSlack = 256;      // x 2137B  = 534KB
	int maxTileSegsSlack = 4096;     // x 136B   = 544KB
	
	// Check for errors.

	if (!m_colorBuffer)
	{	fail("CudaRaster: Surfaces not set!");	}

	if (!m_module)
	{	fail("CudaRaster: Pixel pipe not set!");	}

	if (!m_vertexBuffer)
	{	fail("CudaRaster: Vertex buffer not set!");	}

	if (!m_indexBuffer)
	{	fail("CudaRaster: Index buffer not set!");	}

	// Select batch size for BinRaster and estimate buffer sizes.
	{
		int roundSize = CR_BIN_WARPS * 32;
		int minBatches = CR_BIN_STREAMS_SIZE * 2;
		int maxRounds = 32;

		m_binBatchSize = clamp(m_numTris / (roundSize * minBatches), 1, maxRounds) * roundSize;
		m_maxSubtris = max(m_maxSubtris, m_numTris + maxSubtrisSlack);
		m_maxBinSegs = max(m_maxBinSegs, max(m_numBins * CR_BIN_STREAMS_SIZE, (m_numTris - 1) / CR_BIN_SEG_SIZE + 1) + maxBinSegsSlack);
		m_maxTileSegs = max(m_maxTileSegs, max(m_numTiles, (m_numTris - 1) / CR_TILE_SEG_SIZE + 1) + maxTileSegsSlack);
	}

	// Retry until successful.

	for (;;)
	{
		// Allocate buffers.

		if (m_maxSubtris > CR_MAXSUBTRIS_SIZE)
		{	fail("CudaRaster: CR_MAXSUBTRIS_SIZE exceeded!");	}

		resizeDiscard(m_triSubtris, m_maxSubtris * sizeof(U8));
		resizeDiscard(m_triHeader, m_maxSubtris * sizeof(CRTriangleHeader));
		resizeDiscard(m_triData, m_maxSubtris * sizeof(CRTriangleData));

		resizeDiscard(m_binSegData, m_maxBinSegs * CR_BIN_SEG_SIZE * sizeof(S32));
		resizeDiscard(m_binSegNext, m_maxBinSegs * sizeof(S32));
		resizeDiscard(m_binSegCount, m_maxBinSegs * sizeof(S32));

		resizeDiscard(m_tileSegData, m_maxTileSegs * CR_TILE_SEG_SIZE * sizeof(S32));
		resizeDiscard(m_tileSegNext, m_maxTileSegs * sizeof(S32));
		resizeDiscard(m_tileSegCount, m_maxTileSegs * sizeof(S32));

		launchStages();

		//// No profiling => launch stages.
		//if (m_pipeSpec.profilingMode == ProfilingMode_Default)
		//	launchStages();
		//// Otherwise => setup data buffer, and launch multiple times.
		//else
		//{
		//	int numCounters = FW_ARRAY_SIZE(g_profCounters);
		//	int numTimers = FW_ARRAY_SIZE(g_profTimers);
		//	int totalWarps = m_numSMs * max(CR_BIN_WARPS, CR_COARSE_WARPS, m_numFineWarps);
		//	int bytesPerWarp = max(numCounters * 64 * (int)sizeof(S64), numTimers * 32 * (int)sizeof(U32));

		//	m_profData.resizeDiscard(totalWarps * bytesPerWarp);
		//	m_profData.clear(0);
		//	*(CUdeviceptr*)m_module->getGlobal("c_profData").getMutablePtrDiscard() = m_profData.getMutableCudaPtr();

		//	int numLaunches = (m_pipeSpec.profilingMode == ProfilingMode_Timers) ? numTimers : 1;
		//	for (int i = 0; i < numLaunches; i++)
		//	{
		//		*(S32*)m_module->getGlobal("c_profLaunchIdx").getMutablePtrDiscard() = i;
		//		launchStages();
		//	}
		//}
		// No overflows => done.
		  
		CRAtomics atomics;
   		getGlobal<CRAtomics>(m_module, "g_crAtomics", atomics);

		if (atomics.numSubtris <= m_maxSubtris && atomics.numBinSegs <= m_maxBinSegs && atomics.numTileSegs <= m_maxTileSegs)
			break;

		// Grow buffers and retry.

		m_maxSubtris = max(m_maxSubtris, atomics.numSubtris + maxSubtrisSlack);
		m_maxBinSegs = max(m_maxBinSegs, atomics.numBinSegs + maxBinSegsSlack);
		m_maxTileSegs = max(m_maxTileSegs, atomics.numTileSegs + maxTileSegsSlack);
	}

	m_deferredClear = false;
}

//------------------------------------------------------------------------

CudaRaster::Stats CudaRaster::getStats(void)
{
	Stats stats;
	memset(&stats, 0, sizeof(Stats));
	succeed(cuCtxSynchronize());

	cuEventElapsedTime(&stats.setupTime, m_evSetupBegin, m_evBinBegin);
	cuEventElapsedTime(&stats.binTime, m_evBinBegin, m_evCoarseBegin);
	cuEventElapsedTime(&stats.coarseTime, m_evCoarseBegin, m_evFineBegin);
	cuEventElapsedTime(&stats.fineTime, m_evFineBegin, m_evFineEnd);

	stats.setupTime *= 1.0e-3f;
	stats.binTime *= 1.0e-3f;
	stats.coarseTime *= 1.0e-3f;
	stats.fineTime *= 1.0e-3f;
	return stats;
}

//------------------------------------------------------------------------

//String CudaRaster::getProfilingInfo(void)
//{
//	String s;
//	s += "\n";
//
//	if (!m_module)
//	{
//		s += "Pixel pipe not set!\n";
//	}
//
//	// ProfilingMode_Default.
//
//	if (m_pipeSpec.profilingMode == ProfilingMode_Default)
//	{
//		Stats               stats = getStats();
//		const CRAtomics&    atomics = *(const CRAtomics*)m_module->getGlobal("g_crAtomics").getPtr();
//		F32                 pctCoef = 100.0f / (stats.setupTime + stats.binTime + stats.coarseTime + stats.fineTime);
//		int                 bytesPerSubtri = (int)(sizeof(U8)+sizeof(CRTriangleHeader)+sizeof(CRTriangleData));
//		int                 bytesPerBinSeg = (CR_BIN_SEG_SIZE + 2) * (int)sizeof(S32);
//		int                 bytesPerTileSeg = (CR_TILE_SEG_SIZE + 2) * (int)sizeof(S32);
//
//		s += "ProfilingMode_Default\n";
//		s += "---------------------\n";
//		s += "\n";
//		s += sprintf("%-16s%.3f ms (%.0f%%)\n", "triangleSetup", stats.setupTime * 1.0e3f, stats.setupTime * pctCoef);
//		s += sprintf("%-16s%.3f ms (%.0f%%)\n", "binRaster", stats.binTime * 1.0e3f, stats.binTime * pctCoef);
//		s += sprintf("%-16s%.3f ms (%.0f%%)\n", "coarseRaster", stats.coarseTime * 1.0e3f, stats.coarseTime * pctCoef);
//		s += sprintf("%-16s%.3f ms (%.0f%%)\n", "fineRaster", stats.fineTime * 1.0e3f, stats.fineTime * pctCoef);
//		s += "\n";
//		s += sprintf("%-16s%-10d(%.1f MB)\n", "numSubtris", atomics.numSubtris, (F32)(atomics.numSubtris * bytesPerSubtri) * exp2(-20));
//		s += sprintf("%-16s%-10d(%.1f MB)\n", "numBinSegs", atomics.numBinSegs, (F32)(atomics.numBinSegs * bytesPerBinSeg) * exp2(-20));
//		s += sprintf("%-16s%-10d(%.1f MB)\n", "numTileSegs", atomics.numTileSegs, (F32)(atomics.numTileSegs * bytesPerTileSeg) * exp2(-20));
//	}
//
//	// ProfilingMode_Counters.
//
//	else if (m_pipeSpec.profilingMode == ProfilingMode_Counters)
//	{
//		const S64*  counterPtr = (const S64*)m_profData.getPtr();
//		int         numCounters = FW_ARRAY_SIZE(g_profCounters);
//		int         numWarps = (int)m_profData.getSize() / (numCounters * 64 * (int)sizeof(S64));
//
//		s += "ProfilingMode_Counters\n";
//		s += "----------------------\n";
//		s += "\n";
//
//		for (int i = 0; i < numCounters; i++)
//		{
//			S64 num = 0;
//			S64 denom = 0;
//			for (int j = 0; j < numWarps; j++)
//			for (int k = 0; k < 32; k++)
//			{
//				int idx = (j * numCounters + i) * 64 + k;
//				num += counterPtr[idx + 0];
//				denom += counterPtr[idx + 32];
//			}
//			s += printf(g_profCounters[i].format, (F64)num / max((F64)denom, 1.0));
//		}
//	}
//
//	// ProfilingMode_Timers.
//
//	else if (m_pipeSpec.profilingMode == ProfilingMode_Timers)
//	{
//		const U32*  timerPtr = (const U32*)m_profData.getPtr();
//		int         numTimers = FW_ARRAY_SIZE(g_profTimers);
//		int         numWarps = (int)m_profData.getSize() / (numTimers * 32 * (int)sizeof(U32));
//
//		Array<F64> timers;
//		for (int i = 0; i < numTimers; i++)
//		{
//			U64 launchTotal = 0;
//			for (int j = 0; j < numWarps; j++)
//			{
//				U32 warpTotal = 0;
//				for (int k = 0; k < 32; k++)
//					warpTotal += timerPtr[(j * numTimers + i) * 32 + k];
//				launchTotal += warpTotal;
//			}
//			timers.add((F64)launchTotal);
//		}
//		timers.add(0.0);
//
//		s += "ProfilingMode_Timers\n";
//		s += "--------------------\n";
//		s += "\n";
//		for (int i = 0; i < numTimers; i++)
//			s += printf(g_profTimers[i].format, timers[i] / max(timers[g_profTimers[i].parent], 1.0) * 100.0);
//	}
//
//	else
//	{
//		s += "Invalid profiling mode!\n";
//	}
//
//	s += "\n";
//	return s;
//}

//------------------------------------------------------------------------

void CudaRaster::launchStages(void)
{
	// Set parameters.
	{
		MutableVar<CRParams> var(m_module, "c_crParams");
		CRParams& p = var.get();

		p.numTris = m_numTris;
		p.vertexBuffer = (CUdeviceptr)(m_vertexBuffer->address + m_vertexOfs);
		p.indexBuffer = (CUdeviceptr)(m_indexBuffer->address + m_indexOfs);

		p.viewportWidth = m_viewportSize.x;
		p.viewportHeight = m_viewportSize.y;
		p.widthPixels = m_sizePixels.x;
		p.heightPixels = m_sizePixels.y;

		p.widthBins = m_sizeBins.x;
		p.heightBins = m_sizeBins.y;
		p.numBins = m_numBins;

		p.widthTiles = m_sizeTiles.x;
		p.heightTiles = m_sizeTiles.y;
		p.numTiles = m_numTiles;

		p.binBatchSize = m_binBatchSize;

		p.deferredClear = (m_deferredClear) ? 1 : 0;
		p.clearColor = m_clearColor;
		p.clearDepth = m_clearDepth;

		p.maxSubtris = m_maxSubtris;
		p.triSubtris = m_triSubtris.address;
		p.triHeader = m_triHeader.address;
		p.triData = m_triData.address;

		p.maxBinSegs = m_maxBinSegs;
		p.binFirstSeg = m_binFirstSeg.address;
		p.binTotal = m_binTotal.address;
		p.binSegData = m_binSegData.address;
		p.binSegNext = m_binSegNext.address;
		p.binSegCount = m_binSegCount.address;

		p.maxTileSegs = m_maxTileSegs;
		p.activeTiles = m_activeTiles.address;
		p.tileFirstSeg = m_tileFirstSeg.address;
		p.tileSegData = m_tileSegData.address;
		p.tileSegNext = m_tileSegNext.address;
		p.tileSegCount = m_tileSegCount.address;
	}

	// Initialize atomics.
	{
		MutableVar<CRAtomics> var(m_module, "g_crAtomics");
		CRAtomics& a = var.get();
		a.numSubtris = m_numTris;
		a.binCounter = 0;
		a.numBinSegs = 0;
		a.coarseCounter = 0;
		a.numTileSegs = 0;
		a.numActiveTiles = 0;
		a.fineCounter = 0;
	}

	// Bind textures and surfaces.

	CUdeviceptr vertexPtr = m_vertexBuffer->address;
	S64 vertexSize = m_vertexBuffer->size;

	setTexRef("t_vertexBuffer", m_module, vertexPtr, vertexSize, CU_AD_FORMAT_FLOAT, 4);
	setTexRef("t_triHeader", m_module, m_triHeader.address, m_triHeader.size, CU_AD_FORMAT_UNSIGNED_INT32, 4);
	setTexRef("t_triData", m_module, m_triData.address, m_triData.size, CU_AD_FORMAT_UNSIGNED_INT32, 4);

	setSurfRef("s_colorBuffer", m_module, m_colorBuffer);
	setSurfRef("s_depthBuffer", m_module, m_depthBuffer);

	// Launch triangleSetup().

	succeed(cuEventRecord(m_evSetupBegin, NULL));

	{
		Vec2i block(32, CR_SETUP_WARPS);
		launchParamlessKernelPreferShared(m_setupKernel, m_numTris, block);
	}
	
	// Launch binRaster().

	succeed(cuEventRecord(m_evBinBegin, NULL));

	{ 
		Vec2i block(32, CR_BIN_WARPS);
		launchParamlessKernelPreferShared(m_binKernel, Vec2i(CR_BIN_STREAMS_SIZE, 1) * block, block);
	}

	////// Launch coarseRaster().

	succeed(cuEventRecord(m_evCoarseBegin, NULL));

	{
		Vec2i block(32, CR_COARSE_WARPS);
		launchParamlessKernelPreferShared(m_coarseKernel, Vec2i(m_numSMs, 1) * block, block);
	}

	////// Launch fineRaster().
	// 
	succeed(cuEventRecord(m_evFineBegin, NULL));

	{
		Vec2i block(32, m_numFineWarps);
		launchParamlessKernelPreferShared(m_fineKernel, Vec2i(m_numSMs, 1) * block, block);
	}

	succeed(cuEventRecord(m_evFineEnd, NULL));
}

//------------------------------------------------------------------------
