


#include <type_traits>
#include <cstdint>
#include <ctime>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include <math/vector.h>

#include <GL/platform/InputHandler.h>

#include <pfm.h>
#include <png.h>

#include "Config.h"
#include "OceanScene.h"


using math::float2;
using math::float3;

namespace
{
	class WaterMesh
	{
		std::vector<float3> _vertices;
		std::vector<std::uint32_t> _indices;
		std::vector<uint32_t> _ranges;
		uint32_t _base_level;
		float _unit_range;

		void createMesh()
		{
			uint32_t start_level = _base_level + static_cast<unsigned int>((_ranges.size() - 1));

			uint32_t verts_per_unit_side = (1 << start_level) + 1;

			int total_range = 0;
			for (int i = static_cast<int>(_ranges.size()) - 1; i >= 0; i--)
			{
				int local_range = _ranges[i];
				total_range += local_range;

				for (int j = -total_range; j < total_range; j++)
				{
					float y_off = j * _unit_range;

					for (int k = -total_range; k < total_range; k++)
					{
						float x_off = k * _unit_range;

						if (j >= (-total_range + local_range) && j < (total_range - local_range) && k >= (-total_range + local_range) && k < (total_range - local_range))
						{
							continue;
						}

						uint32_t index_base = static_cast<unsigned int>(_vertices.size());

						int32_t x_tangent = 0, y_tangent = 0;
						if (j >= (-total_range + local_range - 1) && j <= (total_range - local_range) && k >= (-total_range + local_range - 1) && k <= (total_range - local_range))
						{
							if (j == (-total_range + local_range - 1))
							{
								y_tangent = 1;
							}
							else if (j == (total_range - local_range))
							{
								y_tangent = -1;
							}
							if (k == (-total_range + local_range - 1))
							{
								x_tangent = 1;
							}
							else if (k == (total_range - local_range))
							{
								x_tangent = -1;
							}
						}

						int32_t x_index_of_double = -1, y_index_of_double = -1;

						if (i < _ranges.size() - 1 && (x_tangent + y_tangent) && !(x_tangent * y_tangent))
						{
							int32_t y_indic = (x_tangent + 1) / 2;
							int32_t x_indic = (y_tangent + 1) / 2;
							x_index_of_double = (x_tangent ? (y_indic * verts_per_unit_side - y_indic) : -1);
							y_index_of_double = (y_tangent ? (x_indic * verts_per_unit_side - x_indic) : -1);
						}

						for (unsigned int y = 0; y < verts_per_unit_side; y++)
						{
							float vert_y = y_off + (y * _unit_range) / (verts_per_unit_side - 1);

							uint32_t x_verts_per_unit_side = verts_per_unit_side;

							if (y == y_index_of_double)
							{
								x_verts_per_unit_side = (2 * verts_per_unit_side - 1);
							}

							for (unsigned int x = 0; x < x_verts_per_unit_side; x++)
							{
								float vert_x = x_off + (x * _unit_range) / (x_verts_per_unit_side - 1);
								_vertices.push_back(math::float3(vert_x, 0, vert_y));

								if (x == x_index_of_double && y < (verts_per_unit_side - 1))
								{
									_vertices.push_back(math::float3(vert_x, 0, vert_y + (0.5f * _unit_range) / (verts_per_unit_side - 1)));
								}
							}
						}

						uint32_t offset = 0;
						uint32_t y0_offset = 0;
						uint32_t y1_offset = 0;

						for (unsigned int y = 0; y < (verts_per_unit_side - 1); y++)
						{
							for (unsigned int x = 0; x < (verts_per_unit_side - 1); x++)
							{

								if (x_index_of_double == 0)
								{
									y0_offset = y + 1;
									y1_offset = std::min(verts_per_unit_side - 1, y + 2);
								}
								else if (x_index_of_double != -1)
								{
									y0_offset = y;
									y1_offset = y + 1;
								}

								if (y == y_index_of_double)
								{
									uint32_t temp = verts_per_unit_side - 1;

									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + (temp + x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + (temp + x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (2 * x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (2 * x + 0));

									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (2 * x + 1));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + (temp + x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (2 * x + 2));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (2 * x + 1));

									offset = temp;
								}
								else if (y == (y_index_of_double - 1))
								{
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + (2 * x + 1));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + (2 * x + 2));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (x + 0));

									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + (2 * x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + (2 * x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + (x + 0));
								}
								else if (x == x_index_of_double)
								{
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y1_offset + (x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset - 1 + (x + 0));

									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y0_offset + (x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y1_offset + (x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 0));
								}
								else if (x == x_index_of_double - 1)
								{
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y1_offset + (x + 0));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 2));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 0));

									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 2));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y1_offset + (x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y1_offset + (x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (x + 2));
								}
								else
								{
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y1_offset + (offset + x + 0));
									_indices.push_back(index_base + (y + 1) * verts_per_unit_side + y1_offset + (offset + x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (offset + x + 1));
									_indices.push_back(index_base + (y + 0) * verts_per_unit_side + y0_offset + (offset + x + 0));
								}
							}
						}
					}
				}

				verts_per_unit_side = (verts_per_unit_side >> 1) + 1;
			}
		}

	public:
		std::vector<float3>& positions() { return _vertices; }
		std::vector<std::uint32_t>& indices() { return _indices; }

		WaterMesh(float unit_range, uint32_t* ranges, uint32_t num_levels, uint32_t base_level) : _base_level(base_level), _unit_range(unit_range)
		{
			_ranges.resize(num_levels);
			memcpy(_ranges.data(), ranges, sizeof(uint32_t) * num_levels);
			createMesh();
		}
	};
}

OceanScene::OceanScene(const Config& config)
    : img(0, 0), normal_map(0, 0)
{
	paused = 0;

	current_time = 0;
	boop_index = 0;
	base_waves = 5;
	max_boop_waves = 2;

	math::float2 mean_direction = {1, 0};

	float median_length = 4;
	float median_amplitude = 0.15;
	float G = 0.000001f;
	float Q = 0.15;
	float play = 0.6f;

	waves.resize(base_waves + max_boop_waves);

	srand(static_cast<unsigned int>(time(0)));
	for (unsigned int i = 0; i < base_waves; i++)
	{
		float PI = 3.14159265358979f;

		float r, r2;
		r = rand() / static_cast<float>(RAND_MAX);
		float ratio = (0.5f + 1.5f * r);
		float L = ratio * median_length;
		float w = (2 * PI) / L;
		float S = sqrt(G * (2 * PI) / L);
		float A = ratio * median_amplitude;

		r = rand() / static_cast<float>(RAND_MAX);
		r2 = rand() / static_cast<float>(RAND_MAX);

		math::float2 dir(mean_direction.x + (-play + r * 2 * play), mean_direction.y + (-play + r2 * 2 * play));

		waves[i] = {Q, A, w, S * w, normalize(dir)};
	}

	auto& cfg = config.loadConfig("ocean");

	paused = cfg.loadInt("paused", paused);
	base_waves = cfg.loadInt("base_waves", base_waves);
	max_boop_waves = cfg.loadInt("max_boop_waves", max_boop_waves);
	boop_index = cfg.loadInt("boop_index", boop_index);
	current_time = cfg.loadFloat("current_time", current_time);

	camera_pos.x = cfg.loadFloat("camera_pos_x", camera_pos.x);
	camera_pos.y = cfg.loadFloat("camera_pos_y", camera_pos.y);
	camera_pos.z = cfg.loadFloat("camera_pos_z", camera_pos.z);

	center_pos.x = cfg.loadFloat("center_pos_x", center_pos.x);
	center_pos.y = cfg.loadFloat("center_pos_y", center_pos.y);

	waves.resize(base_waves + max_boop_waves);

	int boops = std::min(max_boop_waves, boop_index);

	for (unsigned int i = 0; i < base_waves + boops; i++)
	{
		std::string post = std::to_string(i);
		waves[i].A = cfg.loadFloat((std::string("A") + post).c_str(), waves[i].A);
		waves[i].Q = cfg.loadFloat((std::string("Q") + post).c_str(), waves[i].Q);
		waves[i].phi = cfg.loadFloat((std::string("phi") + post).c_str(), waves[i].phi);
		waves[i].w = cfg.loadFloat((std::string("w") + post).c_str(), waves[i].w);
		waves[i].D.x = cfg.loadFloat((std::string("D_x") + post).c_str(), waves[i].D.x);
		waves[i].D.y = cfg.loadFloat((std::string("D_y") + post).c_str(), waves[i].D.y);
	}
}

void OceanScene::createImage(const char* fname)
{
	img = PFM::loadRGB32F(fname);
}

std::string textureLayerName(const std::string& basename, int i)
{
	std::ostringstream fn;
	fn << basename << "." << i << ".png";
	return fn.str();
}

template <typename InputIt, typename OutputIt>
OutputIt flipCopy(OutputIt dest, const InputIt src, size_t w, size_t h)
{
	for (int r = 0; r < h; ++r)
	{
		auto s = src + (h - 1 - r) * w;

		std::copy(s, s + w, dest);
		dest += w;
	}

	return dest;
}

void OceanScene::createNormalMap(const char* fname, bool single)
{
	//normal_map = PNG::loadRGBA8(fname);

	std::unique_ptr<std::uint32_t[]> texdata;
	unsigned int texw = 256U;
	unsigned int texh = 256U;

	std::string basename(fname);

	auto layer_image_0 = textureLayerName(basename, 0);

	{
		auto size = PNG::readSize(layer_image_0.c_str());
		texw = static_cast<unsigned int>(std::get<0>(size));
		texh = static_cast<unsigned int>(std::get<1>(size));
	}

	if (single)
	{
		texdata = std::make_unique<std::uint32_t[]>(texw * texh);
	}
	else
	{
		texdata = std::make_unique<std::uint32_t[]>(texw * texh * 4);
	}

	int texlevels = 0;
	auto teximage = &texdata[0];

	static const int pattern_size = 6;

	for (unsigned int w = texw, h = texh; w > 1 || h > 1; w = std::max(w / 2U, 1U), h = std::max(h / 2U, 1U))
	{
		if (single && texlevels > 0)
		{
			break;
		}

		auto layer_image = textureLayerName(basename, texlevels);

		auto png = PNG::loadRGBA8(layer_image.c_str());

		if (width(png) != w || height(png) != h)
			throw std::runtime_error("mip level image dimension mismatch");

		teximage = flipCopy(teximage, data(png), w, h);

		++texlevels;
	}

	texture = std::tuple<std::unique_ptr<uint32_t[]>, size_t, size_t, int>(std::move(texdata), texw, texh, texlevels);
}

void OceanScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		//water demo
		//std::vector<uint32_t> ranges = { 1, 2, 2, 2, 3, 1, 1};
		//WaterMesh mesh(20.f, ranges.data(), ranges.size(), 0);
		std::vector<uint32_t> ranges = {5, 2, 3, 1, 1, 1};
		WaterMesh mesh(20.f, ranges.data(), static_cast<unsigned int>(ranges.size()), 0);

		//checker demo
		//std::vector<uint32_t> ranges = { 1, 1, 1, 1, 1 };
		//WaterMesh mesh(20.f, ranges.data(), ranges.size(), 1);

		createImage("assets/sky20.pfm");

		createNormalMap("assets/water_normal");
		//createNormalMap("assets/checker2");
		//createNormalMap("assets/checker3", true);

		material = resource_ptr<Material>(renderer->createOceanMaterial(data(img), width(img), height(img), std::get<0>(texture).get(), std::get<1>(texture), std::get<2>(texture), std::get<3>(texture)));
		geometry = resource_ptr<Geometry>(renderer->createOceanGeometry(&mesh.positions()[0].x, size(mesh.positions()), &mesh.indices()[0], size(mesh.indices())));

		std::cout << "generated ocean has " << size(mesh.positions()) << " vertices and " << size(mesh.indices()) << " indices.\n";

		if (!material || !geometry)
			throw std::runtime_error("renderer cannot support this scene type");
	}
	else
	{
		geometry.reset();
		material.reset();
	}
}

void OceanScene::update(Camera::UniformBuffer& buff)
{
	if (!paused)
	{
		buffer = buff;

		camera_pos = buff.position;

		current_time += 20.0f;

		center_pos = camera_pos.xz();
	}
}

void OceanScene::draw(RendereringContext* context) const
{
	context->setLight(math::float3(0.0f, 10.0f, 0.0f), math::float3(1.0f, 1.0f, 1.0f));

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
	                                           0.0f, 1.0f, 0.0f, 0.0f,
	                                           0.0f, 0.0f, 1.0f, 0.0f));

	context->setUniformf(0, current_time);

	int actual_waves = base_waves + std::min(boop_index, max_boop_waves);

	context->setUniformf(1, static_cast<float>(actual_waves));

	//turns off waves
	//context->setUniformf(1, 0);

	context->setUniformf(2, center_pos.x);
	context->setUniformf(3, center_pos.y);

	for (int i = 0; i < actual_waves; i++)
	{
		uint32_t off = 4 + i * 6;
		const Wave& w = waves[i];
		context->setUniformf(off + 0, w.Q);
		context->setUniformf(off + 1, w.A);
		context->setUniformf(off + 2, w.w);
		context->setUniformf(off + 3, w.phi);
		context->setUniformf(off + 4, w.D.x);
		context->setUniformf(off + 5, w.D.y);
	}

	//                                    this is really bad!
	material->draw(geometry.get(), adaptive ? 1 : 0, wireframe ? 1 : 0);
}

void OceanScene::handleButton(GL::platform::Key c)
{
	switch (c)
	{
	case GL::platform::Key::C_B:
		{
			int index = base_waves + (boop_index % max_boop_waves);
			boop_index++;

			float median_length = 2;
			float median_amplitude = 8.15;
			float G = 0.00001f;

			float r = 1.f;
			float ratio = (0.5f + 1.5f * r);
			float L = ratio * median_length;
			float w = (2 * math::constants<float>::pi()) / L;
			float S = sqrt(G * (2 * math::constants<float>::pi()) / L);
			float A = ratio * median_amplitude;

			math::float4 on_far(0, 0, 1, 1);
			math::float4 on_near(0, 0, 0, 1);

			on_far = buffer.PV_inv * on_far;
			on_near = buffer.PV_inv * on_near;

			math::float2 pos = center_pos + 15.f * normalize(math::float2(on_far.x / on_far.w - on_near.x / on_near.w, on_far.z / on_far.w - on_near.z / on_near.w));

			Wave wave = {-current_time, A, w, S * w, pos};

			waves[index] = wave;
		}
		break;

	case GL::platform::Key::C_P:
		paused = !paused;
		break;

	case GL::platform::Key::C_W:
		wireframe = !wireframe;
		break;

	case GL::platform::Key::C_A:
		adaptive = !adaptive;
		break;
	}
}

void OceanScene::save(Config& config) const
{
	auto& cfg = config.loadConfig("ocean");

	cfg.saveInt("paused", paused);
	cfg.saveInt("base_waves", base_waves);
	cfg.saveInt("max_boop_waves", max_boop_waves);
	cfg.saveInt("boop_index", boop_index);

	cfg.saveFloat("current_time", current_time);

	cfg.saveFloat("camera_pos_x", camera_pos.x);
	cfg.saveFloat("camera_pos_y", camera_pos.y);
	cfg.saveFloat("camera_pos_z", camera_pos.z);

	cfg.saveFloat("center_pos_x", center_pos.x);
	cfg.saveFloat("center_pos_y", center_pos.y);

	int boops = std::min(max_boop_waves, boop_index);

	for (unsigned int i = 0; i < base_waves + boops; i++)
	{
		std::string post = std::to_string(i);
		cfg.saveFloat((std::string("A") + post).c_str(), waves[i].A);
		cfg.saveFloat((std::string("Q") + post).c_str(), waves[i].Q);
		cfg.saveFloat((std::string("phi") + post).c_str(), waves[i].phi);
		cfg.saveFloat((std::string("w") + post).c_str(), waves[i].w);
		cfg.saveFloat((std::string("D_x") + post).c_str(), waves[i].D.x);
		cfg.saveFloat((std::string("D_y") + post).c_str(), waves[i].D.y);
	}
}
