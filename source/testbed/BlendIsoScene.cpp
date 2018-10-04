


#include <cstdint>
#include <stdexcept>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <math/vector.h>

#include "BlendIsoScene.h"


void BlendIsoScene::attachOBJ(const char* fname, math::float4 color)
{
	std::ifstream in(fname);
	std::string line;

	std::vector<math::float3> vertices;
	std::vector<math::float3> normals;

	std::map<math::uint3, int, cmpuint3> vertex_map;

	while (std::getline(in, line))
	{
		char c;
		std::stringstream ss;
		ss << line;
		ss >> c;

		if (c == 'v')
		{
			ss >> c;
			if (c == 'n')
			{
				math::float3 n;
				ss >> n.x >> n.y >> n.z;
				normals.push_back(n);
			}
			else if (c == 't')
			{

			}
			else
			{
				math::float3 v;
				ss >> v.x >> v.y >> v.z;
				vertices.push_back(v);
			}
		}
		else if (c == 'f')
		{
			indices_.push_back({ 0,0,0 });

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
				{   v_spec.x = std::atoi(subs[0].c_str()) - 1; }
				if (subs[1].length())
				{   }
				if (subs[2].length())
				{   v_spec.z = std::atoi(subs[2].c_str()) - 1;	}

				if (vertex_map.find(v_spec) == vertex_map.end())
				{
					vertex_map[v_spec] = static_cast<int>(final_vertices_.size());

					Vertex v;
					v.position = vertices[v_spec.x];
					v.normal = normals[v_spec.z];
					v.color = color;

					final_vertices_.push_back(v);
				}

				indices_.back()[s] = vertex_map[v_spec];
			}
		}
	}
}

BlendIsoScene::BlendIsoScene()
{
	//std::vector<const char*> files = { "assets/s_carp0.62.obj" , "assets/s_carp0.42.obj", "assets/s_carp0.15.obj" };
	//std::vector<math::float4> colors = { { 0.0f, 0.0f, 1.0f, 1.0f }, { 1.0, 0.7, 0, 0.95f }, { 1, 0, 0, 0.4f } };

	//std::vector<const char*> files = { "assets/s_piggy_bottom.obj", "assets/s_piggy.obj","assets/s_piggy_coins.obj" };
	//std::vector<math::float4> colors = { { 0.5f, 0.5f, 0, 0.8f }, { 0.0f, 0.7f, 0.7f, 0.85f }, { 1.f, 1.0, 0.0, 0.4f } };

	std::vector<const char*> files = { "assets/s_vismale0.75.obj", "assets/s_vismale0.32.obj", "assets/s_vismale0.25.obj" };
	std::vector<math::float4> colors = { { 0.4f, 0.4, 0.4, 1.0f },{ 1.f, 1.0, 0.0, 0.8f },{ 0.0f, 0.7f, 0.7f, 0.5f } };

	//std::vector<const char*> files = { "assets/s_engine0.7.obj", "assets/s_engine0.4.obj"};
	//std::vector<math::float4> colors = { { 1, 0, 0, 1.0f },{ 0, 1.0, 0, 0.4f }};

	for (int i = 0; i < files.size(); i++)
	{
		attachOBJ(files[i], colors[i]);
	}
}

void BlendIsoScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		material.reset(renderer->createLitMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));
		geometry.reset(renderer->createIsoBlend((float*)final_vertices_.data(), static_cast<unsigned int>(final_vertices_.size()), (uint32_t*)indices_.data(), static_cast<unsigned int>(indices_.size())));

		if (!material || !geometry)
			throw std::runtime_error("renderer cannot support this scene type");
	}
	else
	{
		geometry.reset();
		material.reset();
	}
}

void BlendIsoScene::draw(RendereringContext* context) const
{
	context->clearColorBuffer(1, 1, 1, 1);
	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
	                                           0.0f, 1.0f, 0.0f, 0.0f,
	                                           0.0f, 0.0f, 1.0f, 0.0f));
	material->draw(geometry.get());
}
