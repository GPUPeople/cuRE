


#include <cstdint>
#include <vector>
#include <streambuf>
#include <sstream>
#include <iostream>

#include <math/vector.h>

#include "io.h"
#include "SceneBuilder.h"

#include "binscene.h"


namespace
{
	const size_t MaterialNameLength = 256;
	const size_t TextureNameLength = 256;
	const size_t TextureFilenameLength = 1024;

	class memory_istreambuf : public std::basic_streambuf<char>
	{
	public:
		memory_istreambuf(const char* buffer, size_t length)
		{
			char* b = const_cast<char*>(buffer);
			setg(b, b, b + length);
		}
	};

}

namespace binscene
{
	void read(SceneBuilder& builder, const char* begin, size_t length)
	{
		memory_istreambuf b(begin, length);
		std::istream in(&b);

		std::vector<vertex> vertices;
		std::vector<std::uint32_t> indices;
		std::vector<surface> surfaces;
		std::vector<material> materials;
		std::vector<texture> textures;

		std::uint32_t num_vertices;
		io::read(in, num_vertices);
		vertices.resize(num_vertices);
		io::read(in, &vertices[0], num_vertices);


		std::uint32_t num_indices;
		io::read(in, num_indices);
		indices.resize(num_indices);
		io::read(in, &indices[0], num_indices);

		std::uint32_t num_surfaces;
		io::read(in, num_surfaces);
		surfaces.reserve(num_surfaces);
		for (std::uint32_t i = 0; i < num_surfaces; ++i)
		{
			std::uint32_t start;
			std::uint32_t primitive_type;
			std::uint32_t num_indices;
			int matId;

			io::read(in, primitive_type);
			io::read(in, start);
			io::read(in, num_indices);
			io::read(in, matId);

			std::stringstream sname;
			sname << "surface" << i;
			surfaces.emplace_back(sname.str().c_str(), sname.str().size(), static_cast<PrimitiveType>(primitive_type), start, num_indices, matId);
		}


		std::uint32_t num_materials;
		io::read(in, num_materials);
		materials.reserve(num_materials);
		char mat_name_buf[MaterialNameLength+1];
		for (std::uint32_t i = 0; i < num_materials; ++i)
		{
			math::float3 ambient, diffuse;
			math::float4 specular;
			float alpha;
			int texId;

			io::read(in, ambient);
			io::read(in, diffuse);
			io::read(in, specular);
			io::read(in, alpha);
			io::read(in, texId);

			std::uint32_t name_length;
			io::read(in, name_length);
			name_length = std::min<std::uint32_t>(name_length, MaterialNameLength);
			in.read(mat_name_buf, MaterialNameLength);
			mat_name_buf[name_length] = '\0';

			materials.emplace_back(mat_name_buf, ambient, diffuse, specular, alpha, texId);
		}

		std::uint32_t num_textures;
		io::read(in, num_textures);
		textures.reserve(num_textures);
		char tex_name_buf[TextureNameLength + 1];
		char tex_fname_buf[TextureFilenameLength + 1];
		for (std::uint32_t i = 0; i < num_textures; ++i)
		{
			std::uint32_t name_length, fname_length;

			io::read(in, name_length);
			name_length = std::min<std::uint32_t>(name_length, TextureNameLength);
			in.read(tex_name_buf, TextureNameLength);
			tex_name_buf[name_length] = '\0';

			io::read(in, fname_length);
			fname_length = std::min<std::uint32_t>(fname_length, TextureFilenameLength);
			in.read(tex_fname_buf, TextureFilenameLength);
			tex_fname_buf[fname_length] = '\0';

			textures.emplace_back(tex_name_buf, tex_fname_buf);
		}

		for (auto & tex : textures)
			builder.addTexture(tex.name.c_str(), tex.fname.c_str());
		for (auto & mat : materials)
			builder.addMaterial(mat.name.c_str(), mat.ambient, mat.diffuse, mat.specular, mat.alpha, mat.texId >= 0 ? textures[mat.texId].name.c_str() : nullptr);
		for (auto & surface : surfaces)
		{
			std::vector<std::uint32_t> thisIds(&indices[surface.start], (&indices[surface.start]) + surface.num_indices);
			builder.addSurface(surface.primitive_type, std::move(thisIds), surface.name.c_str(), surface.name.size(), surface.matId >= 0 ? materials[surface.matId].name.c_str() : nullptr);
		}
		builder.addVertices(std::move(vertices));
	}


	void write(std::ostream& file, const vertex* vertices, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, const surface* surfaces, size_t num_surfaces, const material* materials, size_t num_materials, const texture* textures, size_t num_textures)
	{
		std::string nothingness(std::max(TextureFilenameLength, std::max(MaterialNameLength, TextureNameLength)), '\0');

		io::write(file, static_cast<std::uint32_t>(num_vertices));
		io::write(file, vertices, num_vertices);
		

		io::write(file, static_cast<std::uint32_t>(num_indices)); 
		io::write(file, indices, num_indices);

		io::write(file, static_cast<std::uint32_t>(num_surfaces));
		for (const surface* surface = surfaces; surface < surfaces + num_surfaces; ++surface)
		{
			io::write(file, static_cast<std::uint32_t>(surface->primitive_type));
			io::write(file, surface->start);
			io::write(file, surface->num_indices);
			io::write(file, surface->matId);
		}

		io::write(file, static_cast<std::uint32_t>(num_materials));

		for (const material* material = materials; material < materials + num_materials; ++material)
		{
			io::write(file, material->ambient);
			io::write(file, material->diffuse);
			io::write(file, material->specular);
			io::write(file, material->alpha);
			io::write(file, material->texId);
			io::write(file, static_cast<std::uint32_t>(material->name.size()));
			file.write(material->name.c_str(), std::min(material->name.size(), MaterialNameLength));
			file.write(nothingness.c_str(), MaterialNameLength - std::min(material->name.size(), MaterialNameLength));
		}

		io::write(file, static_cast<std::uint32_t>(num_textures));
		for (const texture* texture = textures; texture < textures + num_textures; ++texture)
		{
			io::write(file, static_cast<std::uint32_t>(texture->name.size()));
			file.write(texture->name.c_str(), std::min(texture->name.size(),TextureNameLength));
			file.write(nothingness.c_str(), TextureNameLength - std::min(texture->name.size(), TextureNameLength));

			io::write(file, static_cast<std::uint32_t>(texture->fname.size()));
			file.write(texture->fname.c_str(), std::min(texture->fname.size(), TextureFilenameLength));
			file.write(nothingness.c_str(), TextureFilenameLength - std::min(texture->fname.size(), TextureFilenameLength));
		}
	}
}
