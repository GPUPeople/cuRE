


#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "io.h"

#include "Renderer.h"
#include "Display.h"

#include "EyeCandyScene.h"
#include "Config.h"


struct CombinedVertex
{
	math::float4 position;
	math::float3 normal;
	math::float3 color;
	float ao;
};

struct IndexedTriangle
{
	uint32_t v1_;
	uint32_t v2_;
	uint32_t v3_;
};



EyeCandyScene::EyeCandyScene(const char* scene, const Config& config, Display& display, ShaderType shader_type)
	: shader_type(shader_type)
{
	auto& cfg = config.loadConfig("eyecandyscene");
	use_drawcalls_ = (cfg.loadInt("separate_surfaces", 0) == 1);

	std::ifstream file(scene, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open scenefile");

	//#0 resolution
	file.read((char*)&width, sizeof(uint32_t));
	file.read((char*)&height, sizeof(uint32_t));
	//#1 #vertices
	file.read((char*)&num_vertices, sizeof(uint32_t));
	//#2 vertex data (#vertices)
	std::unique_ptr<CombinedVertex[]> temp_vertices = std::make_unique<CombinedVertex[]>(num_vertices);
	file.read((char*)temp_vertices.get(), sizeof(CombinedVertex) * num_vertices);

	vertices = std::make_unique<GPUVertex[]>(num_vertices);
	for (unsigned int i = 0; i < num_vertices; i++)
	{
		//vertices[i] = { temp_vertices[i].position};
		//vertices[i] = { temp_vertices[i].position, math::float4(temp_vertices[i].normal, 0)};
		vertices[i] = {temp_vertices[i].position, math::float4(temp_vertices[i].normal, 0), math::float4(temp_vertices[i].color, 0)};
	}

	//#3 #meshes
	file.read((char*)&num_meshes, sizeof(uint32_t));
	//#4 colors (#meshes)
	std::vector<math::float3> colors(num_meshes);
	file.read((char*)colors.data(), sizeof(math::float3) * num_meshes);
	//#5 ranges (#meshes)
	ranges_.resize(num_meshes);
	file.read((char*)ranges_.data(), sizeof(Range) * num_meshes);
	//#6 #triangles
	file.read((char*)&num_triangles, sizeof(uint32_t));
	//#7 triangle data (#triangles)
	std::vector<IndexedTriangle> temp_triangles(num_triangles);
	file.read((char*)temp_triangles.data(), sizeof(IndexedTriangle) * num_triangles);

	triangles = std::make_unique<math::uint3[]>(num_triangles * 2);
	for (unsigned int i = 0; i < num_triangles; i++)
	{
		triangles[i] = math::uint3(temp_triangles[i].v1_, temp_triangles[i].v2_, temp_triangles[i].v3_);
		triangles[num_triangles + i] = math::uint3(temp_triangles[i].v3_, temp_triangles[i].v2_, temp_triangles[i].v1_);
	}
	num_triangles *= 2;

	//num_triangles = num_triangles * 0.779552f;
	//num_triangles += 1;

	//triangle_colors = std::make_unique<math::float3[]>(num_triangles);
	//for (int i = 0; i < num_meshes; i++)
	//{
	//    uint32_t from = ranges[i].from;
	//    uint32_t count = ranges[i].count;
	//    for (int j = 0; j < count; j++)
	//    {   triangle_colors[from + j] = colors[i];  }
	//}

	std::cout << "EyeCandy scene has\n\t"
	          << num_vertices << " vertices\n\t"
	          << num_triangles << " triangles\n\n";

	if (file.fail())
		throw std::runtime_error("error reading scenefile");

	display.resizeWindow(width, height);
}

void EyeCandyScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		switch (shader_type)
		{
		case ShaderType::VERTEX_HEAVY:
			material.reset(renderer->createVertexHeavyEyeCandyMaterial(16));
			break;

		case ShaderType::FRAGMENT_HEAVY:
			material.reset(renderer->createFragmentHeavyEyeCandyMaterial(16));
			break;

		default:
			material.reset(renderer->createEyeCandyMaterial());
		}

		geometry.reset(renderer->createEyeCandyGeometry(&vertices[0].pos.x, num_vertices, &triangles[0].x, &triangle_colors[0].x, num_triangles));
	}
	else
	{
		material.reset();
		geometry.reset();
	}
}

void EyeCandyScene::draw(RendereringContext* context) const
{
	if (!use_drawcalls_)
	{
		//material->draw(geometry.get(), 0, num_triangles * 3);
		material->draw(geometry.get());
	}
	else
	{
		for (int i = 0; i < ranges_.size(); i++)
		{
			material->draw(geometry.get(), ranges_[i].from * 3, ranges_[i].count*3);
			material->draw(geometry.get(), ((num_triangles/2) + ranges_[i].from) * 3, ranges_[i].count * 3);
		}
	}
}
