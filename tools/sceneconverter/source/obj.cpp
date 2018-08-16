


#include <cstdint>
#include <limits>
#include <cctype>
#include <algorithm>
#include <string>
#include <streambuf>
#include <sstream>
#include <fstream>
#include <iostream>
#include <exception>
#include <memory>
#include <tuple>
#include <vector>
#include <unordered_map>

#include <math/vector.h>

#include "SceneBuilder.h"

#include "obj.h"


namespace
{
	class memory_istreambuf : public std::basic_streambuf<char>
	{
	public: 
		memory_istreambuf(const char* buffer, size_t length)
		{
			char* b = const_cast<char*>(buffer);
			setg(b, b, b + length);
		}

		std::streampos seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
		{
			return std::streampos(gptr() - eback());
		}

		std::streampos  seekpos(pos_type off, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
		{
			return std::streampos(gptr() - eback());
		}
	};

	std::string getIdentifier(std::istream& in, bool walkoverline = false)
	{
		char c;
		std::stringstream identifier;

		while (in && isspace(c = in.get()) && (c != '\n' || walkoverline) );
		if (c == '\n' || walkoverline)
		{ 
			in.putback('\n');
			return "";
		}

		identifier << c;
		while (in && !isspace(c = in.get()))
			identifier << c;
		in.putback(c);

		return identifier.str();
	}

	void read_material(SceneBuilder& builder, const std::string& fname, const std::string& name)
	{
		std::cout << "\rreading " << fname << " " << name << "\n";
		std::ifstream fin(fname.c_str(), std::ios::binary);

		if (!fin)
			throw std::runtime_error("unable to open file " + fname);

		fin.seekg(0, std::ios::end);

		size_t file_size = fin.tellg();
		std::unique_ptr<char[]> buffer(new char[file_size]);

		fin.seekg(0, std::ios::beg);
		fin.read(&buffer[0], file_size);

		memory_istreambuf b(buffer.get(), file_size);
		std::istream in(&b);

		std::string matname = name, tex;
		math::float3 ambient(0.0f), diffuse(1.0f);
		math::float4 specular(0.0f, 0.0f, 0.0f, 1.0f);
		float alpha = 1.0f;

		bool hasmaterial = false;
		while (true)
		{
			while (in && std::isspace(in.peek()))
				in.get();

			if (!in)
				break;

			std::string identifier = getIdentifier(in);
			if (identifier.empty())
				continue;
			else if (identifier[0] == '#' || identifier[0] == std::char_traits<char>::eof())
				while (in && in.get() != '\n');
			else if (identifier.compare("newmtl") == 0)
			{
				if (hasmaterial)
				{
					builder.addMaterial(matname.c_str(), ambient, diffuse, specular, alpha, tex.empty() ? nullptr : tex.c_str());
					matname = name; tex.clear();
					ambient = math::float3(0.0f);
					diffuse = math::float3(1.0f);
					specular = math::float4(0.0f, 0.0f, 0.0f, 1.0f);
					alpha = 1.0f;
				}


				matname = getIdentifier(in);
				while (in && in.get() != '\n');
				hasmaterial = true;
			}
			else if (identifier.compare("Ka") == 0)
				in >> ambient.x >> ambient.y >> ambient.z;
			else if (identifier.compare("Kd") == 0)
				in >> diffuse.x >> diffuse.y >> diffuse.z;
			else if (identifier.compare("Ks") == 0)
				in >> specular.x >> specular.y >> specular.z;
			else if (identifier.compare("Ns") == 0)
				in >> specular.w;
			else if (identifier.compare("Tr") == 0 || identifier.compare("d") == 0)
				in >> alpha;
			else if (identifier.compare("map_Kd") == 0)
			{
				tex = getIdentifier(in);
				//if (!builder.hasTexture(tex.c_str()))
				builder.addTexture(tex.c_str(), tex.c_str());
			}
			else
			{
				std::cout << "\rWARNING: unsupported material parameter: \"" << identifier << "\"\n";
				while (in && in.get() != '\n');
			}
		}

		if (hasmaterial)
			builder.addMaterial(matname.c_str(), ambient, diffuse, specular, alpha, tex.empty() ? nullptr : tex.c_str());
	}
}

namespace obj
{
	void read(SceneBuilder& builder, const char* begin, size_t length)
	{
		using namespace math;

		std::vector<float3> v;
		std::vector<float3> vn;
		std::vector<float2> vt;

		struct vertex_key_t
		{
			std::uint32_t v;
			std::uint32_t n;
			std::uint32_t t;

			vertex_key_t(std::uint32_t v, std::uint32_t n, std::uint32_t t)
				: v(v), n(n), t(t)
			{
			}

			bool operator ==(const vertex_key_t& key) const
			{
				return v == key.v && n == key.n && t == key.t;
			}
		};

		struct vertex_hash_t
		{
			size_t operator ()(const vertex_key_t& key) const
			{
				return (key.v << 16) ^ (key.n << 8) ^ key.t;
			}
		};

		std::unordered_map<vertex_key_t, std::uint32_t, vertex_hash_t> vertex_map;

		std::vector<vertex> vertices;
		std::vector<std::uint32_t> indices;

		memory_istreambuf b(begin, length);
		std::istream in(&b);


		std::string group_name;
		std::string current_material;
		auto primitive_type = PrimitiveType::TRIANGLES;

		std::cout << "\n";
		int row = 0;
		while (true)
		{
			if (++row % 1000 == 0)
			{
				size_t cur = in.tellg();
				float p = cur * 1000 / length *0.1f;
				std::cout << '\r' << p << "%";
			}
				
			while (in && std::isspace(in.peek()))
				in.get();

			if (in)
			{
				char c = in.get();

				switch (c)
				{
				case 'v':
				{
					if (std::isspace(in.peek()))
					{
						float3 p;
						in >> p.x >> p.y >> p.z;
						p.z = -p.z;
						v.push_back(p);
					}
					else
					{
						char c2 = in.get();

						switch (c2)
						{
						case 'n':
						{
							float3 n;
							in >> n.x >> n.y >> n.z;
							n.z = -n.z;
							vn.push_back(n);
							break;
						}
						case 't':
						{
							float2 t;
							in >> t.x >> t.y;
							vt.push_back(t);

							while (in && in.peek() != '\n' && std::isspace(in.peek()))
								in.get();

							float t3;
							if (in.peek() != '\n')
							{
								in >> t3;
								if (in.fail())
								{
									in.clear();
									std::cout << "\rWARNING(ln " << row << "): additional content after 2D texture coordinate" << '\n';
								}
								else if (t3 != 0)
								{
									static bool warned = false;
									if (!warned)
										std::cout << "\rWARNING(ln " << row << "): 3D texture coordinates not supported" << '\n';
									warned = true;
								}
									
							}
						}
						}
					}
					break;
				}

				case 'f':
				{
					auto face_start = indices.size();

					while (in && in.peek() != '\n')
					{
						int vi, ni = -1, ti = -1;

						in >> vi;

						if (vi < 0)
							vi += static_cast<int>(v.size());
						else
							--vi;
						
						if (in.peek() == '/')
						{
							in.get();

							in >> ti;

							if (in.fail())
								in.clear();
							else
							{
								if (ti < 0)
									ti += static_cast<int>(vt.size());
								else
									--ti;
							}

							if (in.peek() == '/')
							{
								in.get();

								in >> ni;

								if (in.fail())
									in.clear();
								else
								{
									if (ni < 0)
										ni += static_cast<int>(vn.size());
									else
										--ni;
								}
							}
						}


						vertex_key_t key(vi, ni, ti);

						std::uint32_t index;

						auto found_v = vertex_map.find(key);
						if (found_v != end(vertex_map))
							index = found_v->second;
						else
						{
							float3 p = v[vi];
							float3 n = ni < 0 ? float3(0.0f, 0.0f, 0.0f) : vn[ni];
							float2 t = ti < 0 ? float2(0.0f, 0.0f) : vt[ti];

							index = static_cast<std::uint32_t>(vertices.size());
							vertex_map.insert(std::make_pair(key, index));
							vertices.emplace_back(p, n, t);
						}

						indices.push_back(index);
					}

					std::reverse(std::begin(indices) + face_start, std::end(indices));

					auto num_vertices = indices.size() - face_start;

					switch (num_vertices)
					{
					case 3:
						primitive_type = PrimitiveType::TRIANGLES;
						break;
					case 4:
						primitive_type = PrimitiveType::QUADS;
						break;
					}

					break;
				}

				case 'g':
				{
					if (!indices.empty())
					{
						builder.addSurface(primitive_type, std::move(indices), group_name.empty() ? nullptr : &group_name[0], group_name.length(), current_material.empty() ? nullptr : current_material.c_str());
						indices.clear();
					}
					in >> group_name;
					break;
				}

				case '#':
					// ignore comments
					while (in && in.get() != '\n');
					break;

				case 'm':
				{
					// this could be mtllib mtlfile name
					in.putback('m');
					std::string identifier = getIdentifier(in);
					if (identifier.compare("mtllib") == 0)
					{
						std::string fname, matname;
						if (in.peek() != '\n')
							fname = getIdentifier(in);
						else
						{
							std::cout << "\rWARNING: mtllib instruction without filename" << '\n';
							break;
						}
						if (in.peek() != '\n')
							matname = getIdentifier(in);
						read_material(builder, fname, matname);
						break;
					}
					else
					{ 
						while (identifier.length() > 0)
							in.putback(identifier.back()), identifier.pop_back();
						c = in.get();
						goto UnknownEntry;
					}
					break;
				}

				case 'u':
				{
					// this could be usemtl name
					in.putback('u');
					std::string identifier = getIdentifier(in);
					if (identifier.compare("usemtl") == 0)
					{
						std::string matname = getIdentifier(in);
						if (matname.compare(current_material) != 0)
						{
							if (!indices.empty())
							{
								builder.addSurface(primitive_type, std::move(indices), group_name.empty() ? nullptr : &group_name[0], group_name.length(), current_material.empty() ? nullptr : current_material.c_str());
								indices.clear();
							}
						}
						current_material = matname;
						break;
					}
					else
					{ 
						while (identifier.length() > 0)
							in.putback(identifier.back()), identifier.pop_back();
						goto UnknownEntry;
						break;
					}
				}

				default:
				UnknownEntry:
				{
					std::stringstream unknown;
					bool nontrivial = (!std::isspace(c) && !std::iscntrl(c) && (c != std::char_traits<char>::eof()));
					unknown << c;
					while (in && (c = in.get()) != '\n')
						nontrivial |= (!std::isspace(c) && !std::iscntrl(c) && (c != std::char_traits<char>::eof())),
						unknown << c;
					if (nontrivial)
						std::cout << "\rWARNING (ln " << row << "): unknown entry in obj: \"" << unknown.str() << "\"\n";
					break;
				}
			}
			}
			else
				break;
		}

		builder.addVertices(std::move(vertices));

		if (!indices.empty())
			builder.addSurface(primitive_type, std::move(indices), group_name.empty() ? nullptr : &group_name[0], group_name.length(), current_material.empty() ? nullptr : current_material.c_str());

		std::cout << '\r' << "Obj Read completed\n";
	}

	void write(std::ostream& file, const vertex* vertices, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, const surface* surfaces, size_t num_surfaces, const material* materials, size_t num_materials, const texture* textures, size_t num_textures)
	{
		// TODO: material and texture support

		for (const vertex* v = vertices; v < vertices + num_vertices; ++v)
		{
			file << "v " << v->p.x << ' ' << v->p.y << ' ' << v->p.z << '\n';
		}

		for (const vertex* v = vertices; v < vertices + num_vertices; ++v)
		{
			file << "vn " << v->n.x << ' ' << v->n.y << ' ' << v->n.z << '\n';
		}

		for (const vertex* v = vertices; v < vertices + num_vertices; ++v)
		{
			file << "vt " << v->t.x << ' ' << v->t.y << '\n';
		}

		for (const surface* s = surfaces; s < surfaces + num_surfaces; ++s)
		{
			file << "g " << s->name << '\n';

			for (std::uint32_t i = s->start; i < s->start + s->num_indices; i += 6)
				file << "f " << 1 + indices[i] <<'/' << 1 + indices[i] << '/' << 1 + indices[i] << ' '
				     << 1 + indices[i + 2] <<'/' << 1 + indices[i + 2] << '/' << 1 + indices[i + 2] << ' '
				     << 1 + indices[i + 4] <<'/' << 1 + indices[i + 4] << '/' << 1 + indices[i + 4] << '\n';
		}
	}
}
