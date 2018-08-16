


#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <fstream>

#include "io.h"

#include "Renderer.h"

#include "ClipspaceScene.h"


ClipspaceScene::ClipspaceScene(const char* scenefile, ShaderType shader_type)
	: shader_type(shader_type)
{
	std::ifstream file(scenefile, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open scenefile");

	num_vertices = read<std::uint32_t>(file);

	vertices = std::make_unique<float[]>(2 * num_vertices * 4);

	read(file, &vertices[0], 4 * num_vertices);

	if (file.fail())
		throw std::runtime_error("error reading scenefile");

	for (int i = 0; i < num_vertices; i += 3)
	{
		memcpy(&vertices[4 * (num_vertices + i + 2)], &vertices[4 * (i + 0)], sizeof(float) * 4);
		memcpy(&vertices[4 * (num_vertices + i + 1)], &vertices[4 * (i + 1)], sizeof(float) * 4);
		memcpy(&vertices[4 * (num_vertices + i + 0)], &vertices[4 * (i + 2)], sizeof(float) * 4);
	}

	num_vertices *= 2;

	std::cout << "Clipspace scene has\n\t"
	          << num_vertices / 3 << " faces\n\n";
}


void ClipspaceScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		switch (shader_type)
		{
		case ShaderType::VERTEX_HEAVY:
			material.reset(renderer->createVertexHeavyClipspaceMaterial(8));
			break;

		case ShaderType::VERTEX_SUPER_HEAVY:
			material.reset(renderer->createVertexHeavyClipspaceMaterial(32));
			break;

		case ShaderType::FRAGMENT_HEAVY:
			material.reset(renderer->createFragmentHeavyClipspaceMaterial(8));
			break;

		case ShaderType::FRAGMENT_SUPER_HEAVY:
			material.reset(renderer->createFragmentHeavyClipspaceMaterial(32));
			break;

		default:
			material.reset(renderer->createClipspaceMaterial());
		}

		geometry.reset(renderer->createClipspaceGeometry(&vertices[0], num_vertices));
	}
	else
	{
		material.reset();
		geometry.reset();
	}
}

void ClipspaceScene::draw(RendereringContext* context) const
{
	material->draw(geometry.get());
}
