


#ifndef INCLUDED_PLUGIN_MANAGER
#define INCLUDED_PLUGIN_MANAGER

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include <filesystem>

#include <win32/unique_handle.h>
#include <win32/module.h>

#include "PlugInClient.h"
#include "Renderer.h"


class Config;

class PlugInManager
{
public:
	class INTERFACE Callback
	{
	protected:
		Callback() = default;
		Callback(const Callback&) = default;
		Callback& operator =(const Callback&) = default;
		~Callback() = default;
	public:
		virtual void onRendererPlugInLoaded(const char* name, createRendererFunc create_function) = 0;
		virtual void onRendererPlugInUnloading(createRendererFunc create_function) = 0;
		virtual void onDetach(PlugInManager* plugin_man) = 0;
	};

private:
	PlugInManager(const PlugInManager&) = delete;
	PlugInManager& operator =(const PlugInManager&) = delete;

	class Module : private virtual PlugInClient
	{
	private:
		Win32::unique_hmodule hdll;

		std::vector<std::tuple<std::string, createRendererFunc>> renderers;

		void registerRenderer(const char* name, createRendererFunc create_function);

		Callback* callback;

	public:
		Module(const Module&) = delete;
		Module& operator =(const Module&) = delete;

		Module(Win32::unique_hmodule hdll);
		~Module();

		Module(Module&&);
		Module& operator =(Module&&);

		void attach(Callback* callback);
	};

	std::filesystem::path plugin_dir;
	std::filesystem::path hot_swap_dir;

	std::vector<std::tuple<Module, std::wstring, FILETIME>> modules;

	Callback* callback;

public:
	PlugInManager(const Config& config);

	void loadModule(const wchar_t* name);

	void refreshModules();

	void attach(Callback* callback);

	void save(Config& config) const;
};

#endif  // INCLUDED_PLUGIN_MANAGER
