


#ifndef INCLUDED_NVRTC_PROGRAM
#define INCLUDED_NVRTC_PROGRAM

#pragma once

#include <initializer_list>
#include <memory>
#include <string>

#include <nvrtc.h>

#include <CUDA/unique_handle.h>


namespace NVRTC
{
	struct ProgramDeleter
	{
		void operator ()(nvrtcProgram program) const
		{
			nvrtcDestroyProgram(&program);
		}
	};

	using unique_program = CU::unique_handle<nvrtcProgram, nullptr, ProgramDeleter>;


	unique_program createProgram(const char* source, const char* name, const char* const * headers, int num_headers, const char* const * include_names);


	template <std::size_t N>
	inline nvrtcResult compileProgram(nvrtcProgram prog, const char* (&options)[N])
	{
		return nvrtcCompileProgram(prog, N, options);
	}

	inline nvrtcResult compileProgram(nvrtcProgram prog, std::initializer_list<const char*> options)
	{
		return nvrtcCompileProgram(prog, static_cast<int>(std::end(options) - std::begin(options)), std::begin(options));
	}


	std::string getProgramLog(nvrtcProgram prog);


	struct PTXObject
	{
		std::unique_ptr<char[]> data;
		std::size_t size;
	};

	PTXObject getPTX(nvrtcProgram prog);
}

#endif  // INCLUDED_NVRTC_PROGRAM
