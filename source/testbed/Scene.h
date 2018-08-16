#ifndef INCLUDED_SCENE
#define INCLUDED_SCENE

#pragma once

#include <GL/platform/InputHandler.h>

#include "Resource.h"
#include "Renderer.h"


class Config;

class INTERFACE Scene
{
protected:
	Scene() = default;
	Scene(const Scene&) = default;
	Scene& operator =(const Scene&) = default;
	
public:
	virtual void handleButton(GL::platform::Key c) {}
	virtual void switchRenderer(Renderer* renderer) = 0;
	virtual void draw(RendereringContext* context) const = 0; 
	virtual void update(Camera::UniformBuffer& buff) {};
	virtual void save(Config& config) const = 0;
	virtual ~Scene() = default;
};

#endif  // INCLUDED_SCENE
