


#include <cstring>

#include <algorithm>
#include <string>

#include <iostream>

#include <CUDA/link.h>
#include <CUDA/binary.h>

#include "PipelineModule.h"


namespace CUBIN
{
	extern const char cure;
	extern const char cure_end;

	extern const char geometry_stage;
	extern const char geometry_stage_end;

	extern const char rasterization_stage;
	extern const char rasterization_stage_end;

	extern const char framebuffer;
	extern const char framebuffer_end;

	extern const char instrumentation;
	extern const char instrumentation_end;

    extern const char uniforms;
    extern const char uniforms_end;
}


namespace
{
	class memory_istreambuf : public std::basic_streambuf<char>
	{
	public:
		memory_istreambuf(const char* buffer, size_t length)
		{
			char* b = const_cast<char*>(buffer);
			setg(b, b, b + length);
		}
	};

	struct PipelineKernelInfo
	{
		std::string pipeline_name;
		int rasterizer_count;
		int warps_per_block;
		int virtual_rasterizers;
		std::string symbol_name;
	};

	std::vector<PipelineKernelInfo> findPipelineKernels(const char* gpu_image)
	{
		std::vector<PipelineKernelInfo> kernels;

		for (auto s : CU::readSymbols(static_cast<const char*>(gpu_image)))
		{
			//std::cout << s << std::endl;
			if (std::strncmp(s, ".text._ZN9Pipelines", 19) == 0)
			{
				memory_istreambuf b(s + 19, std::strlen(s + 19));
				std::istream name(&b);

				int name_len;
				name >> name_len;

				std::string pipeline_name(name_len, '\0');
				name.read(&pipeline_name[0], name_len);

				if (name.get() == 'I' && name.get() == 'L' && name.get() == 'j')
				{
					int multiprocessor_count;
					name >> multiprocessor_count;
					if (name.get() == 'E' && name.get() == 'L' && name.get() == 'j')
					{
						int block_per_multiprocessor;
						name >> block_per_multiprocessor;
						if (name.get() == 'E' && name.get() == 'L' && name.get() == 'j')
						{
							int warps_per_block;
							name >> warps_per_block;

							if (name.get() == 'E' && name.get() == 'L' && name.get() == 'j')
							{
								int virtual_rasterizer_count;
								name >> virtual_rasterizer_count;

								if (name.get() == 'E' && name.get() == 'E' && name.get() == 'E' && name.get() == 'v' && name.get() == 'v' && name.get() && name.eof())
									kernels.push_back({ std::move(pipeline_name), multiprocessor_count * block_per_multiprocessor, warps_per_block, virtual_rasterizer_count, s + 6 });
							}
						}
					}
				}
			}
		}

		return kernels;
	}
}

namespace cuRE
{
	PipelineModule::PipelineModule()
	{
		auto linker = CU::createLinker(0U, nullptr, nullptr);

		succeed(cuLinkAddData(linker, CUjitInputType::CU_JIT_INPUT_CUBIN, const_cast<char*>(&CUBIN::cure), &CUBIN::cure_end - &CUBIN::cure, "cure.cuobj", 0U, nullptr, nullptr));
		succeed(cuLinkAddData(linker, CUjitInputType::CU_JIT_INPUT_CUBIN, const_cast<char*>(&CUBIN::geometry_stage), &CUBIN::geometry_stage_end - &CUBIN::geometry_stage, "geometry_stage.cuobj", 0U, nullptr, nullptr));
		succeed(cuLinkAddData(linker, CUjitInputType::CU_JIT_INPUT_CUBIN, const_cast<char*>(&CUBIN::rasterization_stage), &CUBIN::rasterization_stage_end - &CUBIN::rasterization_stage, "rasterization_stage.cuobj", 0U, nullptr, nullptr));
		succeed(cuLinkAddData(linker, CUjitInputType::CU_JIT_INPUT_CUBIN, const_cast<char*>(&CUBIN::framebuffer), &CUBIN::framebuffer_end - &CUBIN::framebuffer, "framebuffer.cuobj", 0U, nullptr, nullptr));
		succeed(cuLinkAddData(linker, CUjitInputType::CU_JIT_INPUT_CUBIN, const_cast<char*>(&CUBIN::instrumentation), &CUBIN::instrumentation_end - &CUBIN::instrumentation, "instrumentation.cuobj", 0U, nullptr, nullptr));
		succeed(cuLinkAddData(linker, CUjitInputType::CU_JIT_INPUT_CUBIN, const_cast<char*>(&CUBIN::uniforms), &CUBIN::uniforms_end - &CUBIN::uniforms, "uniforms.cuobj", 0U, nullptr, nullptr));

		void* gpu_image;
		size_t gpu_image_size;
		succeed(cuLinkComplete(linker, &gpu_image, &gpu_image_size));

		auto kernels = findPipelineKernels(static_cast<const char*>(gpu_image));

		module = CU::loadModule(static_cast<const char*>(gpu_image));
		
		for (auto& k : kernels)
		{
			auto kernel = PipelineKernel(module, k.symbol_name.c_str(), k.rasterizer_count, k.warps_per_block, k.virtual_rasterizers);
			auto o = kernel.computeOccupancy();

			if (o.real() > 0)
			{
				auto found = findKernel(k.pipeline_name.c_str());
				if (found != std::end(pipeline_kernels))
				{
					auto oo = std::get<1>(*found).computeOccupancy();

					if (oo.imag() < o.imag())
						std::get<1>(*found) = std::move(kernel);
				}
				else
				{
					pipeline_kernels.emplace_back(std::move(k.pipeline_name), std::move(kernel));
				}
			}
		}
	}

	CUdeviceptr PipelineModule::getGlobal(const char* name) const
	{
		return CU::getGlobal(module, name);
	}

	CUfunction PipelineModule::getFunction(const char* name) const
	{
		return CU::getFunction(module, name);
	}

	CUtexref PipelineModule::getTextureReference(const char* name) const
	{
		return CU::getTextureReference(module, name);
	}

	CUsurfref PipelineModule::getSurfaceReference(const char* name) const
	{
		return CU::getSurfaceReference(module, name);
	}

	decltype(PipelineModule::pipeline_kernels)::const_iterator PipelineModule::findKernel(const char* name) const
	{
		return std::find_if(std::begin(pipeline_kernels),
		                    std::end(pipeline_kernels),
		                    [name](const decltype(pipeline_kernels)::value_type& kernel) { return std::get<0>(kernel) == name; });
	}

	decltype(PipelineModule::pipeline_kernels)::iterator PipelineModule::findKernel(const char* name)
	{
		return std::find_if(std::begin(pipeline_kernels),
		                    std::end(pipeline_kernels),
		                    [name](const decltype(pipeline_kernels)::value_type& kernel) { return std::get<0>(kernel) == name; });
	}

	PipelineKernel PipelineModule::findPipelineKernel(const char* name) const
	{
		auto found = findKernel(name);
		if (found != std::end(pipeline_kernels))
			return std::get<1>(*found);
		throw std::runtime_error("no pipeline kernel found");
	}
}
