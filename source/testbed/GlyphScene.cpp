


#include <cstdint>
#include <math/vector.h>

#include "GlyphScene.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


struct PosColor
{
	math::float3 pos;
	math::float3 color;
};

void GlyphScene::attachOBJ(const char* fname, math::float4 color, math::float4 offscale, float rot)
{
	std::ifstream in(fname);
	std::string line;

	std::vector<PosColor> vertices;
	std::vector<math::float3> normals;

	std::map<math::uint3, int, cmpuint3> vertex_map;

	math::float2 fake_uv[] = {{0, 0}, {0.5f, 0}, {1, 1}};

	rot = rot * math::constants<float>::pi() / 180.0f;

	math::float3x3 rotmat =
	    {cos(rot), -sin(rot), 0,
	     sin(rot), cos(rot), 0,
	     0, 0, 0};

	while (std::getline(in, line))
	{
		char c;
		std::stringstream ss;
		ss << line;
		ss >> c;

		if (c == 'v')
		{
			if (ss.str().length() > 0)
			{
				c = ss.str()[1];
				if (c == 'n')
				{
					ss.ignore(1);
					math::float3 n;
					ss >> n.x >> n.y >> n.z;
					normals.push_back(n);
				}
				else if (c == 't')
				{
				}
				else
				{
					PosColor pc;

					math::float3 v;
					ss >> v.x >> v.y >> v.z;

					v = rotmat * v;

					v.x = v.x * offscale.z + offscale.x;
					v.y = v.y * offscale.w + offscale.y;

					pc.pos = v;

					ss >> v.x >> v.y >> v.z;

					pc.color = v;

					vertices.push_back(pc);
				}
			}
		}
		else if (c == 'f')
		{
			indices_.push_back({0, 0, 0});

			for (int s = 0; s < 3; s++)
			{
				std::string tf;
				ss >> tf;

				std::vector<std::string> subs(3);
				int found = 0;
				for (int i = 0; i < tf.length(); i++)
				{
					if (tf[i] == '/')
					{
						found++;
					}
					else
					{
						subs[found].append(1, tf[i]);
					}
				}

				math::uint3 v_spec(0, 0, 0);

				if (subs[0].length())
				{
					v_spec.x = std::atoi(subs[0].c_str()) - 1;
				}
				if (subs[1].length())
				{
				}
				if (subs[2].length())
				{
					v_spec.z = std::atoi(subs[2].c_str()) - 1;
				}

				if (vertex_map.find(v_spec) == vertex_map.end())
				{
					vertex_map[v_spec] = static_cast<int>(final_vertices_.size());

					Vertex v;
					v.position = vertices[v_spec.x].pos;
					v.color = color;

					float sign = 0;
					if (vertices[v_spec.x].color == math::float3(0, 1, 0))
					{
						sign = -10;
						v.uvsign = math::float3((v.position.x - offscale.x) / offscale.z, (v.position.y - offscale.y) / offscale.w, sign);
					}
					else
					{
						if (vertices[v_spec.x].color == math::float3(1, 0, 0))
						{
							sign = -1;
						}
						else if (vertices[v_spec.x].color == math::float3(0, 0, 1))
						{
							sign = 1;
						}
						v.uvsign = math::float3(fake_uv[s], sign);
					}

					final_vertices_.push_back(v);
				}

				indices_.back()[s] = vertex_map[v_spec];
			}
		}
	}
}

GlyphScene::GlyphScene()
{
	//std::vector<const char*> files = { "assets/vector_demo/cure.obj" };
	/*std::vector<const char*> files = { "assets/vector_demo/cure_jap.obj", */
	std::vector<const char*> files = {"assets/vector_demo/circle.obj", "assets/vector_demo/quad.obj", "assets/vector_demo/tri.obj", "assets/vector_demo/cure_jap.obj"};
	std::vector<math::float4> colors = {{0.9, 0.4, 0.5, 1}, {0.7, 0.7, 0.4, 1}, {0.4, 0.7, 0.7, 1}, {0.2, 0.2, 0.2, 1}, {1, 1, 1, 1}, {0.4, 0.4, 0.4, 1}};
	std::vector<math::float4> offscales = {{25.0, 5.0, 25, 25}, {-3.2, -5.0, 20, 20}, {25, -17, 25, 20}, {-20, -26, 0.06, 0.06}, {-30, 0, 0.05, 0.05}, {10, -35, 0.05, 0.05}};
	std::vector<float> rots = {0, 0, -12, 0, 0, 0, 0};

	for (int i = 0; i < files.size(); i++)
	{
		attachOBJ(files[i], colors[i], offscales[i], rots[i]);
	}
}

void GlyphScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		/*uint64_t mask = 0b1111000001111000001111000001111000001111100001111100001111100001;*/
		uint64_t mask = 0b1111111111111111111111111111111111111111111111111111111111111111;
		material.reset(renderer->createLitMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));
		geometry.reset(renderer->createGlyphDemo(mask, (float*)final_vertices_.data(), static_cast<unsigned int>(final_vertices_.size()), (uint32_t*)indices_.data(), static_cast<unsigned int>(indices_.size())));
	}
	else
	{
		geometry.reset();
		material.reset();
	}
}

void GlyphScene::draw(RendereringContext* context) const
{
	context->clearColorBuffer(1, 1, 1, 1);
	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
	                                           0.0f, 1.0f, 0.0f, 0.0f,
	                                           0.0f, 0.0f, 1.0f, 0.0f));
	material->draw(geometry.get());
}
