
#ifndef INCLUDED_WATERSCENE
#define INCLUDED_WATERSCENE

#pragma once

#include <math/vector.h>
#include <math/matrix.h>

#include "Resource.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"

#include "resource_ptr.h"
#include <image_tools/include/image.h>
#include <image_tools/include/pfm.h>
#include <image_tools/include/png.h>

struct Wave
{
    float Q, A, w, phi;
    math::float2 D;
};

class WaterScene : public Scene
{
	resource_ptr<Geometry> geometry;
	resource_ptr<Material> material;
    image2D<RGB32F> img;
    image2D<RGBA8> normal_map;

    std::tuple<std::unique_ptr<uint32_t[]>, size_t, size_t, int> texture;

    std::vector<Wave> waves;

    float current_time;

    math::float3 camera_pos;
    math::float2 center_pos;

    uint32_t base_waves;
    uint32_t max_boop_waves;
    uint32_t boop_index;

    Camera::UniformBuffer buffer;

    int paused;

    void createImage(const char* fname);
    void createNormalMap(const char* fname, bool single = false);

public:
    WaterScene(const Config& config);
    WaterScene(const WaterScene&) = delete;
    WaterScene& operator =(const WaterScene&) = delete;

    virtual void handleButton(GL::platform::Key c) override;

	void switchRenderer(Renderer* renderer);
    void update(Camera::UniformBuffer& buff) override;
	void draw(RendereringContext* context) const;

	void save(Config& config) const override;

	static void* operator new(std::size_t size);
	static void operator delete(void* p);
};

#endif  // INCLUDED_SCENE
