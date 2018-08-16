//
//
//
//#include <cstring>
//
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <stdexcept>
//#include <memory>
//
//#include "PerspectiveCamera.h" 
//#include "OrbitalCameraNavigator.h"
//#include "Renderer.h"
//#include <CUDA/context.h>
//#include <CUDA/module.h>
//#include "base/helpling.h"
//
//#include "Shaders.hpp"
//#include "App.hpp"
//#include <vector>
//
//size_t resX = 64;
//size_t resY = 32;
////{ -1, -1, -0.5 }, , { -0.5, 0, -0.5 }
//float3 v_[] = { { 0, -1, -0.5 }, { 1, -1, -0.5 }, { 0.5, 0, -0.5 }, { -1, -1, -0.5 }, { -0.5, 0, -0.5 }, {0, 1, -0.5}};
//int3 t_[] = {  {0,1,2}, {3,0,4}, {4,2,5}};
//
//int num_verts = sizeof(v_)/sizeof(float3); 
//int num_tris = sizeof(t_)/sizeof(int3);
//
//namespace CUBIN
//{   
//	extern const char cudaraster;
//}
//
//int main(int argc, char* argv[])
//{
//	//try
//	//{
//		cuInit(0);
//
//		CU::unique_context context = CU::createContext();
//
//		CU::unique_module module = CU::loadModule(&CUBIN::cudaraster);
//
//		Buffer in_vertices;
//		resizeDiscard(in_vertices, num_verts * sizeof(FW::InputVertex));
//		Buffer out_vertices;
//		resizeDiscard(out_vertices, num_verts * sizeof(FW::ShadedVertex_gouraud));
//		Buffer in_indices;
//		resizeDiscard(in_indices, num_tris * sizeof(int3));
//
//		{
//			MutableMem<FW::InputVertex> vertexmem(in_vertices.address, num_verts);
//			FW::InputVertex* v = vertexmem.get();
//			for(int i = 0; i < num_verts; i++)
//			{	v[i].modelPos = FW::Vec3f(v_[i].x, v_[i].y, v_[i].z);	}
//
//			MutableMem<int3> indexmem(in_indices.address, num_tris);
//			int3* i = indexmem.get();
//			for(int j = 0; j < num_tris; j++)
//			{	i[j] = t_[j];	}
//		}
//
//		CUarray colorarray;
//		CUDA_ARRAY3D_DESCRIPTOR colordesc;
//		colordesc.Flags = 2;
//		colordesc.Depth = 0;
//		colordesc.Width = resX;
//		colordesc.Height = resY;
//		colordesc.NumChannels = 4;
//		colordesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
//		cuArray3DCreate_v2(&colorarray, &colordesc);
//
//		CUarray deptharray;
//		CUDA_ARRAY3D_DESCRIPTOR depthdesc;
//		depthdesc.Flags = 2;
//		depthdesc.Depth = 0;
//		depthdesc.Width = resX;
//		depthdesc.Height = resY;
//		depthdesc.NumChannels = 1;
//		depthdesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
//		succeed(cuArray3DCreate_v2(&deptharray, &depthdesc));
//
//		Buffer dummy_mat, dummy_ind, dummy_vmat, dummy_tmat;
//
//		uchar4 zero = {255,255,255,255};
//		std::vector<uchar4> test_color(resX*resY, zero);
//
//		//for (int y = 0; y < resY; y++)
//		//{
//		//	for (int x = 0; x < resX; x++)
//		//	{	std::cout << (test_color[y*resX + x].x) / 255;	}
//		//	std::cout << std::endl;
//		//}
//
//		CUDA_MEMCPY3D memm;
//		//memm.dstMemoryType = CU_MEMORYTYPE_ARRAY;
//		//memm.dstArray = colorarray;
//		//memm.dstXInBytes = 0;
//		//memm.dstY = 0;
//		//memm.dstZ = 0;
//		//memm.dstLOD = 0;
//
//		//memm.srcMemoryType = CU_MEMORYTYPE_HOST;
//		//memm.srcHost = test_color.data();
//		//memm.srcXInBytes = 0;
//		//memm.srcY = 0;
//		//memm.srcZ = 0;
//		//memm.srcLOD = 0;
//		//memm.srcPitch = resX*sizeof(uchar4);
//		//memm.srcHeight = resY;
//
//		//memm.WidthInBytes = resX*sizeof(uchar4);
//		//memm.Height = resY;
//		//memm.Depth = 1;
//
//		//succeed(cuMemcpy3D_v2(&memm));
//
//		FW::App app(module);
//		app.setData(in_vertices, out_vertices, dummy_mat, in_indices, dummy_vmat, dummy_tmat, num_verts, 0, num_tris);
//		app.setTargets(colorarray, deptharray, resX, resY);
//		app.initPipe();
//		app.render("lol.bmp");
//
//		std::vector<uchar4> color(resX*resY);
//
//		memm.srcMemoryType = CU_MEMORYTYPE_ARRAY;
//		memm.srcArray = colorarray;
//		memm.srcXInBytes = 0;
//		memm.srcY = 0;
//		memm.srcZ = 0;
//		memm.srcLOD = 0;
//
//		memm.dstMemoryType = CU_MEMORYTYPE_HOST;
//		memm.dstHost = color.data();
//		memm.dstXInBytes = 0;
//		memm.dstY = 0;
//		memm.dstZ = 0;
//		memm.dstLOD = 0;
//		memm.dstPitch = resX*sizeof(uchar4);
//		memm.dstHeight = resY;
//
//		memm.WidthInBytes = resX*sizeof(uchar4);
//		memm.Height = resY;
//		memm.Depth = 1;
//		
//		succeed(cuMemcpy3D_v2(&memm));
//		//succeed(cuMemcpyAtoH(color.data(), colorarray, 0, sizeof(uchar4)*resX*resY));
//
//		for(int y = resY-1; y >= 0; y--)
//		{
//			for(int x = 0; x < resX; x++)
//			{	std::cout << (color[y*resX+x].x)/255;	}
//			std::cout << std::endl;
//		}
//	//}
//	//catch (std::exception& e)
//	//{
//	//	std::cout << "error: " << e.what() << std::endl;
//	//	return -1;
//	//}
//	//catch (...)
//	//{
//	//	std::cout << "unknown exception" << std::endl;
//	//	return -128;
//	//}
//	return 0;
//
//	//try
//	//{
//	//	cuInit(0);
//
//	//	CU::unique_context context = CU::createContext();
//
//	//	CU::unique_module module = CU::loadModule(&CUBIN::Renderer);
//
//	//	CUfunction func;
//	//	succeed(cuModuleGetFunction(&func, module, "printTestVariable"));
//
//	//	{	MutableVar<int32_t>(module, "test_variable").get() = 55;	}
//
//	//	succeed(cuFuncSetBlockShape(func, 1, 1, 1));
//
//	//	succeed(cuLaunchGrid(func, 1, 1));
//
//	//	return 0;
//	//}
//	//catch (std::exception& e)
//	//{
//	//	std::cout << "error: " << e.what() << std::endl;
//	//	return -1;
//	//}
//	//catch (...)
//	//{
//	//	std::cout << "unknown exception" << std::endl;
//	//	return -128;
//	//}
//
//
//	//try
//	//{
//	//	Renderer renderer;
//	//	Scene* scene = renderer.getScene();
//	//	math::float3 minp = scene->boundingBox().first;
//	//	math::float3 maxp = scene->boundingBox().second;
//	//	float dist = length(maxp - minp);
//
//	//	PerspectiveCamera camera(math::constants<float>::pi() * 1.0f / 3.0f, 0.0001f*dist, 3.0f*dist);
//	//	OrbitalNavigator navigator(-math::constants<float>::pi() * 0.5f, 0.0f, 0.8f*dist, 0.5f*(minp + maxp));
//
//
//	//	InputHandler input_handler(navigator, renderer, *scene);
//
//	//	camera.attach(&navigator);
//
//	//	renderer.attach(&input_handler);
//	//	renderer.attach(&navigator);
//	//	renderer.attach(&renderer);
//
//	//	renderer.attach(&camera);
//
//	//	GL::platform::run(renderer);
//	//}
//	//catch (std::exception& e)
//	//{
//	//	std::cout << "error: " << e.what() << std::endl;
//	//	return -1;
//	//}
//	//catch (...)
//	//{
//	//	std::cout << "unknown exception" << std::endl;
//	//	return -128;
//	//}
//
//	return 0;
//}
