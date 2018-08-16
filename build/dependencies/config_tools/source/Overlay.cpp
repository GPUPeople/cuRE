


#include <config/error.h>
#include <config/Overlay.h>


namespace config
{
	DatabaseOverlay::DatabaseOverlay(Database& base, std::initializer_list<const Database*> overlays)
		: base(base),
		  overlays(overlays)
	{
		this->overlays.push_back(&base);
	}


	void DatabaseOverlay::traverse(Visitor& visitor) const
	{
		for (auto c = rbegin(overlays); c != rend(overlays); ++c)
			(*c)->traverse(visitor);
	}


	const char* DatabaseOverlay::fetchString(const char* key) const
	{
		for (auto c : overlays)
		{
			const char* value;
			if (c->fetchString(value, key))
				return value;
		}
		throw not_found();
	}

	bool DatabaseOverlay::fetchString(const char*& value, const char* key) const
	{
		for (auto c : overlays)
			if (c->fetchString(value, key))
				return true;
		return false;
	}

	const char* DatabaseOverlay::queryString(const char* key, const char* default_value) const
	{
		for (auto c : overlays)
		{
			const char* value;
			if (c->fetchString(value, key))
				return value;
		}
		return default_value;
	}


	int DatabaseOverlay::fetchInt(const char* key) const
	{
		for (auto c : overlays)
		{
			int value;
			if (c->fetchInt(value, key))
				return value;
		}
		throw not_found();
	}

	bool config::DatabaseOverlay::fetchInt(int& value, const char* key) const
	{
		for (auto c : overlays)
		{
			if (c->fetchInt(value, key))
				return true;
		}
		return false;
	}

	int DatabaseOverlay::queryInt(const char* key, int default_value) const
	{
		for (auto c : overlays)
		{
			int value;
			if (c->fetchInt(value, key))
				return value;
		}
		return default_value;
	}


	float DatabaseOverlay::fetchFloat(const char* key) const
	{
		for (auto c : overlays)
		{
			float value;
			if (c->fetchFloat(value, key))
				return value;
		}
		throw not_found();
	}

	bool config::DatabaseOverlay::fetchFloat(float& value, const char* key) const
	{
		for (auto c : overlays)
		{
			if (c->fetchFloat(value, key))
				return true;
		}
		return false;
	}

	float DatabaseOverlay::queryFloat(const char* key, float default_value) const
	{
		for (auto c : overlays)
		{
			float value;
			if (c->fetchFloat(value, key))
				return value;
		}
		return default_value;
	}


	const Database& DatabaseOverlay::fetchNode(const char* key) const
	{
		for (auto c : overlays)
		{
			const Database* node;
			if (c->fetchNode(node, key))
				return *node;
		}
		throw not_found();
	}

	Database& DatabaseOverlay::fetchNode(const char* key)
	{
		return base.fetchNode(key);
	}

	bool DatabaseOverlay::fetchNode(const Database*& node, const char* key) const
	{
		for (auto c : overlays)
			if (c->fetchNode(node, key))
				return true;
		return false;
	}

	bool DatabaseOverlay::fetchNode(Database*& node, const char* key)
	{
		return base.fetchNode(node, key);
	}

	const Database& DatabaseOverlay::queryNode(const char* key, const Database& default_node) const
	{
		for (auto c : overlays)
		{
			const Database* node;
			if (c->fetchNode(node, key))
				return *node;
		}
		return default_node;
	}

	Database& DatabaseOverlay::queryNode(const char* key, Database& default_node)
	{
		return base.queryNode(key, default_node);
	}

	Database& DatabaseOverlay::openNode(const char* key)
	{
		return base.openNode(key);
	}


	void DatabaseOverlay::storeString(const char* key, const char* value)
	{
		base.storeString(key, value);
	}

	void DatabaseOverlay::storeString(const char* key, const std::string& value)
	{
		base.storeString(key, value);
	}

	void DatabaseOverlay::storeString(const char* key, std::string&& value)
	{
		base.storeString(key, value);
	}

	void DatabaseOverlay::storeInt(const char* key, int value)
	{
		base.storeInt(key, value);
	}

	void DatabaseOverlay::storeFloat(const char* key, float value)
	{
		base.storeFloat(key, value);
	}
}
