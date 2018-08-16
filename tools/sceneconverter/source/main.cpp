


#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <memory>
#include <fstream>
#include <iostream>
#include <string>

#include "Scene.h"

#include "obj.h"
#include "sdkmesh.h"
#include "binscene.h"


namespace
{
	bool argcomp(const char* a, const char* b)
	{
		return std::strcmp(a, b) == 0;
	}

	template <size_t N>
	bool argcomp(const char* arg, const char(&c)[N])
	{
		return std::strncmp(arg, c, N - 1) == 0;
	}

	const char* extension(char* source_file)
	{
		char* end = source_file + std::strlen(source_file);
		char* ext = end;

		while (ext != source_file && *--ext != '.');

		if (ext != source_file)
		{
			std::transform(ext, end, ext, std::tolower);
			return ext;
		}

		return nullptr;
	}

	std::string getTargetFile(const char* source_file, const char* extension)
	{
		const char* ext = std::strrchr(source_file, '.');

		if (ext != nullptr)
		{
			return std::string(source_file, ext) + extension;
		}

		return std::string(source_file) + extension;
	}

	import_func_t* selectImporter(char* source_file)
	{
		const char* ext = extension(source_file);

		if (std::strcmp(ext, ".sdkmesh") == 0)
			return &sdkmesh::read;
		else if (std::strcmp(ext, ".obj") == 0)
			return &obj::read;
		else if (std::strcmp(ext, ".scene") == 0)
			return &binscene::read;
		else
			throw std::runtime_error("unsupported file format");
	}

	export_func_t* selectExporter(char* source_file)
	{
		const char* ext = extension(source_file);

		if (std::strcmp(ext, ".scene") == 0)
			return &binscene::write;
		else if (std::strcmp(ext, ".obj") == 0)
			return &obj::write;
		else
			throw std::runtime_error("unsupported file format");
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "usage: sceneconverter [-o <target-file>] [-nomaterial] [-mergeequalmaterial] {[-f] <source-file>}\n" << std::endl;
		return -1;
	}

	try
	{
		bool nomaterial = false;
		bool mergeequalmaterial = false;
		std::string target_file;
		bool next_frame = false;

		Scene scene;

		for (char** arg = &argv[1]; *arg; ++arg)
		{
			if (argcomp(*arg, "-nomaterial"))
				nomaterial = true;
			else if (argcomp(*arg, "-mergeequalmaterial"))
				mergeequalmaterial = true;
			else if (argcomp(*arg, "-o"))
			{
				target_file = *++arg;
				continue;
			}
			else if (argcomp(*arg, "-f"))
				next_frame = true;
			else
			{
				if (target_file.empty())
					target_file = getTargetFile(*arg, ".scene");

				std::ifstream in(*arg, std::ios::binary);

				if (!in)
					throw std::runtime_error("unable to open file " + std::string(*arg));

				in.seekg(0, std::ios::end);

				size_t file_size = in.tellg();
				auto buffer = std::unique_ptr<char[]>{ new char[file_size] };

				in.seekg(0, std::ios::beg);
				in.read(&buffer[0], file_size);

				import_func_t* import_func = selectImporter(*arg);

				if (next_frame)
				{
					scene.importFrame(import_func, &buffer[0], file_size);
					next_frame = false;
				}
				else
					scene.import(import_func, &buffer[0], file_size);
			}
		}

		export_func_t* export_func = selectExporter(&target_file[0]);

		scene.serialize(&target_file[0], export_func, nomaterial, mergeequalmaterial);
	}
	catch (const std::exception& e)
	{
		std::cout << "ERROR: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cout << "unknown error" << std::endl;
		return -1;
	}

	return 0;
}
