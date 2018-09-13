#!/usr/bin/env python3


import os
import subprocess
import shutil
from pathlib import Path
import argparse


this_dir = Path(__file__).parent.resolve()


def _msbuild(*args, **kwargs):
	p = subprocess.Popen(["msbuild", *args], **kwargs)
	return p.wait()


def _instantiateTemplates(dir):
	for p in os.scandir(dir):
		if p.is_dir():
			_instantiateTemplates(p.path)
		else:
			name, ext = os.path.splitext(p.path)
			if ext == ".template" and not os.path.exists(name):
				print(p.path)
				shutil.copy2(p.path, name)

def _build(project_file, platform, *configurations):
	for configuration in configurations:
		if _msbuild("/m", "/p:Platform={}".format(platform), "/p:Configuration={}".format(configuration), project_file.resolve().as_posix()) != 0:
			raise Exception("build error")

def _buildDependencies():
	deps_dir = this_dir/"build/dependencies"
	_build(deps_dir/"COFF_tools/dotNET/COFF_tools.sln", "Any CPU", "Debug")
	_build(deps_dir/"CUDA_build_tools/embedCUDA/source/embedCUDA.sln", "Any CPU", "Debug")
	_build(deps_dir/"Win32_core_tools/build/vs2017/Win32_core_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"COM_core_tools/build/vs2017/COM_core_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"image_tools/build/vs2017/image_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"CUDA_compiler_tools/build/vs2017/CUDA_compiler_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"CUDA_binary_tools/build/vs2017/CUDA_binary_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"CUDA_core_tools/build/vs2017/CUDA_core_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"CUPTI_core_tools/build/vs2017/CUPTI_core_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"CUDA_graphics_interop_tools/build/vs2017/CUDA_graphics_interop_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"GL_platform_tools/build/vs2017/glcore.vcxproj", "x64", "Debug", "Release", "Debug DLL", "Release DLL")
	_build(deps_dir/"GL_platform_tools/build/vs2017/GL_platform_tools.vcxproj", "x64", "Debug", "Release", "Debug DLL", "Release DLL")
	_build(deps_dir/"config_tools/build/vs2017/config_tools.vcxproj", "x64", "Debug", "Release")
	_build(deps_dir/"GL_core_tools/build/vs2017/GL_core_tools.vcxproj", "x64", "Debug", "Release", "Debug DLL", "Release DLL")
	_build(deps_dir/"GLSL_build_tools/build/vs2017/GLSL_build_tools.sln", "x64", "Debug")


def main(args):
	print("instantiating templates...")
	_instantiateTemplates(this_dir)

	print("building dependencies...")
	_buildDependencies()
	print("...done")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	main(parser.parse_args())
