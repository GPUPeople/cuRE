


#include "error.h"
#include "program.h"


namespace NVRTC
{
	unique_program createProgram(const char* source, const char* name, const char* const * headers, int num_headers, const char* const * include_names)
	{
		nvrtcProgram prog;
		succeed(nvrtcCreateProgram(&prog, source, name, num_headers, headers, include_names));
		return unique_program(prog);
	}

	std::string getProgramLog(nvrtcProgram prog)
	{
		std::size_t log_size;
		succeed(nvrtcGetProgramLogSize(prog, &log_size));

		auto log = std::unique_ptr<char[]> { new char[log_size] };
		succeed(nvrtcGetProgramLog(prog, &log[0]));
		return { log.get(), log_size - 1 };
	}

	PTXObject getPTX(nvrtcProgram prog)
	{
		std::size_t size;
		succeed(nvrtcGetPTXSize(prog, &size));

		PTXObject obj = {
			std::unique_ptr<char[]> { new char[size] },
			size
		};

		succeed(nvrtcGetPTX(prog, &obj.data[0]));

		return obj;
	}
}
