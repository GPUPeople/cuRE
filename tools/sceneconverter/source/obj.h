


#ifndef INCLUDED_OBJ
#define INCLUDED_OBJ

#pragma once

#include <iosfwd>


class SceneBuilder;

namespace obj
{
	void read(SceneBuilder& builder, const char* begin, size_t length);
	void write(std::ostream& file, const vertex* vertices, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, const surface* surfaces, size_t num_surfaces, const material* materials, size_t num_materials, const texture* textures, size_t num_textures);
}

#endif  // INCLUDED_OBJ
