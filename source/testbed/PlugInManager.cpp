


#include <algorithm>
#include <sstream>
#include <iostream>

#include <win32/file.h>

#include <objbase.h>

#include "Config.h"

#include "PlugInManager.h"

using std::filesystem::path;


namespace
{
	path imagePath(HMODULE module)
	{
		DWORD length = MAX_PATH;
		std::wstring p(length, L'\0');

		while (GetModuleFileNameW(module, &p[0], length) == ERROR_INSUFFICIENT_BUFFER)
		{
			length *= 2;
			p.resize(length, L'\0');
		}
		return p;
	}

	FILETIME timestamp(const path& filename)
	{
		const Win32::unique_hfile file(CreateFileW(filename.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0));
		FILETIME timestamp;
		GetFileTime(file, nullptr, nullptr, &timestamp);
		return timestamp;
	}

	constexpr bool ENABLE_HOTSWAPPING = false;

	Win32::unique_hmodule loadDLL(const path& dll_path, const path& hot_swap_dir)
	{
		if constexpr (!ENABLE_HOTSWAPPING)
			return Win32::unique_hmodule(LoadLibraryW(dll_path.wstring().c_str()));

		GUID guid;
		CoCreateGuid(&guid);

		wchar_t buffer[256];
		StringFromGUID2(guid, buffer, 256);

		auto pdb_name = dll_path.filename().replace_extension(".pdb");
		path dest_dll_name = path(buffer).concat(".dll");

		path pdb_src = dll_path.parent_path()/pdb_name;
		path pdb_dest = hot_swap_dir/pdb_name;
		CopyFileW(pdb_src.wstring().c_str(), pdb_dest.wstring().c_str(), TRUE);

		path dll_dest = hot_swap_dir/dest_dll_name;

		if (CopyFileW(dll_path.wstring().c_str(), dll_dest.wstring().c_str(), TRUE))
			return Win32::unique_hmodule(LoadLibraryW(dll_dest.wstring().c_str()));

		return Win32::unique_hmodule();
	}
}


PlugInManager::Module::Module(Win32::unique_hmodule hdll)
	: hdll(std::move(hdll)),
	  callback(nullptr)
{
	auto registerPlugInServer = reinterpret_cast<registerPlugInServerFunc>(GetProcAddress(this->hdll, "registerPlugInServer"));
	if (registerPlugInServer)
		registerPlugInServer(this);
}

PlugInManager::Module::~Module()
{
	if (callback)
		for (auto& r : renderers)
			callback->onRendererPlugInUnloading(std::get<1>(r));
}

PlugInManager::Module::Module(Module&& m)
	: hdll(std::move(m.hdll)),
	  renderers(std::move(m.renderers)),
	  callback(m.callback)
{
	m.callback = nullptr;
}

PlugInManager::Module& PlugInManager::Module::operator =(Module&& m)
{
	using std::swap;
	swap(hdll, m.hdll);
	swap(renderers, m.renderers);
	swap(callback, m.callback);
	return *this;
}

void PlugInManager::Module::registerRenderer(const char* name, createRendererFunc create_function)
{
	renderers.push_back(std::make_tuple(name, create_function));
}

void PlugInManager::Module::attach(Callback* callback)
{
	this->callback = callback;
	if (callback)
		for (auto& r : renderers)
			callback->onRendererPlugInLoaded(std::get<0>(r).c_str(), std::get<1>(r));
}


std::filesystem::path PlugInManager::buildDLLPath(const char* module) const
{
	return (plugin_dir/module).concat(".dll");
}

PlugInManager::PlugInManager(const Config& config)
	: plugin_dir(imagePath(0).parent_path()/"plugins"),
	  hot_swap_dir(plugin_dir/"hot"),
	  callback(nullptr)
{
	if (!exists(hot_swap_dir))
		std::filesystem::create_directories(hot_swap_dir);
}


void PlugInManager::refreshModules()
{
	for (auto& m : modules)
	{
		FILETIME t = timestamp(buildDLLPath(std::get<1>(m).c_str()));
		
		if (CompareFileTime(&t, &std::get<2>(m)) > 0)
			loadModule(std::get<1>(m).c_str());
	}
}

void PlugInManager::loadModule(const char* name)
{
	auto dll_path = buildDLLPath(name);
	Win32::unique_hmodule hdll = loadDLL(dll_path, hot_swap_dir);

	if (hdll)
	{
		FILETIME t = timestamp(dll_path);
		Module m = Module(std::move(hdll));
		m.attach(this->callback);

		auto found = std::find_if(begin(modules), end(modules), [name](const auto& m) { return std::get<1>(m) == name; });

		if (found != end(modules))
		{
			std::get<2>(*found) = t;
			std::get<0>(*found) = std::move(m);
		}
		else
		{
			modules.emplace_back(std::make_tuple(std::move(m), name, t));
		}
	}
	else
		std::wcerr << "WARNING: unable to load module '" << name << "'\n";
}

void PlugInManager::attach(Callback* callback)
{
	if (this->callback)
		this->callback->onDetach(this);
	this->callback = callback;
	for (auto& m : modules)
		std::get<0>(m).attach(callback);
}

void PlugInManager::save(Config& config) const
{
	std::vector<std::string> module_names;

	for (auto&& m : modules)
		module_names.emplace_back(std::get<1>(m));

	config.saveTuple("modules", std::move(module_names));
}
