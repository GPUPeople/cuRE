


#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "io.h"

#include "Renderer.h"
#include "Display.h"

#include "Config.h"
#include "CheckerboardScene.h"


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

struct Range
{
	uint32_t from;
	uint32_t count;
};

CheckerboardScene::CheckerboardScene(const Config& config, Display& display)
{
	auto& cfg = config.loadConfig("checkerboard_rendering_demo");
	const char* scenefile = cfg.loadString("scene", "assets/shogun.candy");
	int quad = cfg.loadInt("quad", 0);
	int fragment = cfg.loadInt("fragment", 0);
	int upsample = cfg.loadInt("upsample", 1);
	type = ((upsample & 0x1U) << 2) | ((quad & 0x1U) << 1) | (fragment & 0x1U);


	std::ifstream file(scenefile, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open scenefile");

	std::ifstream infile(scenefile, std::ios_base::in | std::ios_base::binary);
	//#0 resolution
	infile.read((char*)&width, sizeof(uint32_t));
	infile.read((char*)&height, sizeof(uint32_t));
	//#1 #vertices
	infile.read((char*)&num_vertices, sizeof(uint32_t));
	//#2 vertex data (#vertices)
	std::unique_ptr<CombinedVertex[]> temp_vertices = std::make_unique<CombinedVertex[]>(num_vertices);
	infile.read((char*)temp_vertices.get(), sizeof(CombinedVertex) * num_vertices);

	vertices = std::make_unique<GPUVertex[]>(num_vertices);
	for (unsigned int i = 0; i < num_vertices; i++)
	{
		vertices[i] = {temp_vertices[i].position, math::float4(temp_vertices[i].normal, 0), math::float4(temp_vertices[i].color, 0)};
	}

	//#3 #meshes
	infile.read((char*)&num_meshes, sizeof(uint32_t));
	//#4 colors (#meshes)
	std::vector<math::float3> colors(num_meshes);
	infile.read((char*)colors.data(), sizeof(math::float3) * num_meshes);
	//#5 ranges (#meshes)
	std::vector<Range> ranges(num_meshes);
	infile.read((char*)ranges.data(), sizeof(Range) * num_meshes);
	//#6 #triangles
	infile.read((char*)&num_triangles, sizeof(uint32_t));
	//#7 triangle data (#triangles)
	std::vector<IndexedTriangle> temp_triangles(num_triangles);
	infile.read((char*)temp_triangles.data(), sizeof(IndexedTriangle) * num_triangles);

	triangles = std::make_unique<math::uint3[]>(num_triangles * 2);
	for (unsigned int i = 0; i < num_triangles; i++)
	{
		triangles[i] = math::uint3(temp_triangles[i].v1_, temp_triangles[i].v2_, temp_triangles[i].v3_);
		triangles[num_triangles + i] = math::uint3(temp_triangles[i].v3_, temp_triangles[i].v2_, temp_triangles[i].v1_);
	}
	num_triangles *= 2;

	std::cout << num_triangles << std::endl;

	std::cout << "EyeCandy scene has\n"
	          << "\t" << num_vertices << " vertices\n\n";

	if (file.fail())
		throw std::runtime_error("error reading scenefile");

	display.resizeWindow(width, height);
}

void CheckerboardScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		material.reset(renderer->createEyeCandyMaterial());
		geometry.reset(renderer->createCheckerboardGeometry(type, &vertices[0].pos.x, num_vertices, &triangles[0].x, &triangle_colors[0].x, num_triangles));
	}
	else
	{
		material.reset();
		geometry.reset();
	}
}

void CheckerboardScene::draw(RendereringContext* context) const
{
	if (((type >> 1) & 0x1U) == 0U)
		context->clearColorBufferCheckers(0xFFFFFFFF, 0xFF000000, 0);
	else
		context->clearColorBuffer(1.0f, 1.0f, 1.0f, 1.0f);

	material->draw(geometry.get());
}
