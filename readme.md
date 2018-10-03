
A streaming software graphics pipeline for the GPU. To learn how it works, check out the paper [*A High-Performance Software Graphics Pipeline Architecture for the GPU*][1].


## Overview

The project consists of a main testbed application and various renderer plugins. The testbed provides an environment in which different renderers implemented by renderer plugins can be tested and compared. It can draw various kinds of test scenes and offers a basic user interface.

Renderer plugins implement a common interface wich the testbed uses to draw its test scenes. There are currently four renderer plugins: `GLRenderer`, `FreePipe`, `CUDARaster`, and `cuRE`. `GLRenderer` implements a reference renderer using OpenGL. `FreePipe` implements the approach by Lui et al. [[2010]][3]. `CUDARaster` implements the approach of Laine and Karras [[2011]][2]. `cuRE` implements our approach [[Kenzel et al. 2018]][1].

Experiments can be set up via command line arguments and configuration files.
Run
```
testbed --config <config-file>
```
to launch the testbed with the configuration specified in `<config-file>`. Use the `--scene <scene-name>` and `--device <device-id>` options to select the test scene and which CUDA device to run on (note that this does not influence which GPU is used for OpenGL rendering as this choice is up to the graphics driver). The `--renderer <plugin-name>` argument can be used to force a specific renderer plugin to be used. `testbed -h` will print a list of all available options.

Use one of the three default configuration files `all_sm35.cfg`, `all_sm52.cfg`, or `all_sm61.cfg` to start the testbed using all four plugins for the respective GPU architecture. If no configuration file is specified, `config.cfg` will be used. If the configuration file does not exist, an empty configuration will be used. Note that `testbed` will overwrite the configuration file to save the current program state upon exit.

### Controls

  * left mouse:     drag to turn camera
  * middle mouse:   drag to pan camera
  * right mouse:    drag to zoom
  * `[Backspace]`   reset camera
  * `[Tab]`         cycle through renderers (combine with `[Shift]` to cycle backwards)
  * `[F8]`          take screenshot


## How to build

Only Windows 10 (64-Bit) is currently supported. Building the project requires
  * Visual Studio 2017 (15.7+),
  * Windows SDK 10.0.17763.0,
  * CUDA Toolkit 10.0, and
  * Python 3.6+.

To initialize the build system and build dependencies, run `setup.py` from a Visual Studio 2017 Command Prompt. Once this is done, you can open `build/vs2017/cure.sln`.

The solution comes with build configurations targeting the `sm_35`, `sm_52`, and `sm_61` CUDA architectures. Note that there is two kinds of debug configuration, the `Debug_smXX` versions which run debug host with optimized device code, and the `Debug_smXX_d` versions which run debug builds of host and device code.


### cuRE

The cuRE renderer is compiled not just for a specific target architecture, but for a specific launch configuration on the target hardware. This allows us to avoid unnecessary runtime-overhead in the scheduler. However, as a consequence, the parameters of the launch configuration must be provided as compile-time constants.

A list of launch configurations for which to compile is defined in the configuration header `source/cure/pipeline/config.h` line 13:
```cpp
using PipelineConfigs = PipelineConfigList <
	//GTX_780
	//GTX_780_Ti
	//GTX_780_Ti_dynamic
	//GTX_TITAN
	//GTX_960
	//GTX_970
	//GTX_980
	//GTX_980_Ti
	//GTX_980_Ti_dynamic
	//GTX_TITAN_X
	//GTX_1060
	//GTX_1060_dynamic
	//GTX_1070
	//GTX_1070_Ti
	GTX_1080
	//GTX_1080_dynamic
	//GTX_1080_Ti
	//GTX_1080_Ti_dynamic
	//TITAN_X
	//TITAN_X_dynamic
	//TITAN_Xp
>;
```
It is vital that you select an appropriate launch configuration for your GPU prior to compilation. The GPU name identifiers here are simply aliases for a specific
```cpp
PipelineConfig<num_multiprocessors, num_blocks_per_multiprocessor, num_warps_per_block>
```
Add your own in `source/cure/pipeline/gpu_configs.h` or place a `PipelineConfig` directly in `PipelineConfigList<   >`.

> Note: In principle, the `PipelineConfigList` can contain multiple `PipelineConfig` elements. The cuRE plugin will simply contain an instance of the cuRE pipeline kernels for each `PipelineConfig`. At runtime, all pipeline kernel instances are enumerated in the order in which they happen to appear in the CUDA binary symbol table. The first kernel that fits onto the device we are running on is selected to run. Since the kernels have to be compiled once for each `PipelineConfig`, it is generally a good idea to only activate the `PipelineConfig` one currently intends to run to avoid unnecessarily long build times.


#### A word on debugging

Due to the complexity of the device code, it is recommended to avoid running device debug builds unless absolutely necessary as they will be extremely slow (if they compile at all; they have a tendency to exceed resource limits).

The cuRE pipeline is based on a megakernel design. To avoid issues like system crashes and freezes, it is recommended to run a pipeline built for a launch configuration with less multiprocessors than the GPU actually being used has when debugging the GPU code (e.g. using Nsight).


### FreePipe

The `FreePipe` plugin configuration header in `source/FreePipe/config.h` comes with a number of constants that have to do with tweaking the vertex processing stage. There should generally be no need to change these constants. More information concerning their meaning can be found in [Kenzel et al. [2018b]][4]. The only relevant constant in this header is `DEPTH_TEST` which turns depth testing on or off.


### GLRenderer

The `GLRenderer` plugin also comes with a config header in `source/GLRenderer/config.h` which should be mostly self-explanatory. The only option that should need explainig is `FRAGMENT_SHADER_INTERLOCK`. If `FRAGMENT_SHADER_INTERLOCK` is set to `true`, the renderer will use fragment shaders that perform serialized framebuffer access using the `GL_NV_fragment_shader_interlock` extension.


[1]: https://www.tugraz.at/institute/icg/research/team-steinberger/research-projects/gpu-rendering-pipelines/cure/
[2]: http://research.nvidia.com/publication/high-performance-software-rasterization-gpus
[3]: https://dl.acm.org/citation.cfm?id=1730817
[4]: https://www.tugraz.at/institute/icg/research/team-steinberger/research-projects/gpu-rendering-pipelines/vertexreuse/
