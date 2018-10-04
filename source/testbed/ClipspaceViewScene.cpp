


#include <cstdint>
#include <stdexcept>
#include <fstream>

#include "io.h"

#include "Renderer.h"

#include "ClipspaceViewScene.h"


ClipspaceViewScene::ClipspaceViewScene(const char* scenefile)
{
	std::ifstream file(scenefile, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open scenefile");


	num_vertices = read<std::uint32_t>(file);

	auto file_vertices = std::make_unique<float[]>(num_vertices * 4);

	read(file, &file_vertices[0], 4 * num_vertices);


	vertices = std::make_unique<float[]>(num_vertices * 3);
	normals = std::make_unique<float[]>(num_vertices * 3);
	texcoords = std::make_unique<float[]>(num_vertices * 3);
	indices = std::make_unique<std::uint32_t[]>(num_vertices);

	for (int i = 0; i < num_vertices; ++i)
	{
		vertices[3 * i] = file_vertices[4 * i];
		vertices[3 * i + 1] = file_vertices[4 * i + 1];
		vertices[3 * i + 2] = file_vertices[4 * i + 3];

		normals[3 * i] = 0.0f;
		normals[3 * i + 1] = 0.0f;
		normals[3 * i + 2] = 0.0f;

		texcoords[2 * i] = 0.0f;
		texcoords[2 * i + 1] = 0.0f;

		indices[i] = i;
	}

	if (file.fail())
		throw std::runtime_error("error reading scenefile");
}


void ClipspaceViewScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		material.reset(renderer->createLitMaterial(math::float4(0.6f, 0.6f, 0.6f, 1.0f)));
		geometry.reset(renderer->createIndexedTriangles(&vertices[0], &normals[0], &texcoords[0], num_vertices, &indices[0], num_vertices));

		if (!material || !geometry)
			throw std::runtime_error("renderer cannot support this scene type");
	}
	else
	{
		material.reset();
		geometry.reset();
	}
}

void ClipspaceViewScene::draw(RendereringContext* context) const
{
	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
		                                        0.0f, 1.0f, 0.0f, 0.0f,
		                                        0.0f, 0.0f, 1.0f, 0.0f));
	material->draw(geometry.get());
}
