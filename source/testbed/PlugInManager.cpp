


#include <algorithm>
#include <sstream>
#include <iostream>

#include <win32/unicode.h>
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

	Win32::unique_hmodule loadDLL(const std::wstring& name, const path& plugin_dir, const path& hot_swap_dir)
	{
		path src_dll_name = name + L".dll";
		path dll_src = plugin_dir / src_dll_name;

		if constexpr (!ENABLE_HOTSWAPPING)
			return Win32::unique_hmodule(LoadLibraryW(dll_src.wstring().c_str()));

		GUID guid;
		CoCreateGuid(&guid);

		wchar_t buffer[256];
		StringFromGUID2(guid, buffer, 256);

		path pdb_name = name + L".pdb";
		path dest_dll_name = std::wstring(buffer) + L".dll";

		path pdb_src = plugin_dir / pdb_name;
		path pdb_dest = hot_swap_dir / pdb_name;
		CopyFileW(pdb_src.wstring().c_str(), pdb_dest.wstring().c_str(), TRUE);

		path dll_dest = hot_swap_dir / dest_dll_name;

		if (CopyFileW(dll_src.wstring().c_str(), dll_dest.wstring().c_str(), TRUE))
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


PlugInManager::PlugInManager(const Config& config)
	: plugin_dir(imagePath(0).parent_path() / path(L"plugins")),
	  hot_swap_dir(plugin_dir / path(L"hot")),
	  callback(nullptr)
{
	if (!exists(hot_swap_dir))
		std::filesystem::create_directories(hot_swap_dir);

	auto modules = config.loadTuple("modules", {});

	for (auto&& m : modules)
		loadModule(widen(m).c_str());
}


void PlugInManager::refreshModules()
{
	for (auto& m : modules)
	{
		FILETIME t = timestamp(plugin_dir / path(std::get<1>(m) + L".dll"));
		
		if (CompareFileTime(&t, &std::get<2>(m)) > 0)
			loadModule(std::get<1>(m).c_str());
	}
}

void PlugInManager::loadModule(const wchar_t* name)
{
	Win32::unique_hmodule hdll = loadDLL(name, plugin_dir, hot_swap_dir);

	if (hdll)
	{
		FILETIME t = timestamp(plugin_dir / path(std::wstring(name) + L".dll"));
		Module m = Module(std::move(hdll));
		m.attach(this->callback);

		auto found = std::find_if(begin(modules), end(modules), [name](const decltype(modules)::value_type& m) { return std::get<1>(m) == name; });

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
		module_names.emplace_back(narrow(std::get<1>(m)));

	config.saveTuple("modules", std::move(module_names));
}
