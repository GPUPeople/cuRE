


#include <config/error.h>
#include <config/InMemoryDatabase.h>


namespace config
{
	InMemoryDatabase::ChildNode::ChildNode()
		: node(std::make_unique<InMemoryDatabase>())
	{
	}

	void InMemoryDatabase::traverse(Visitor& visitor) const
	{
		for (const auto& v : string_values)
			visitor.visitString(v.first.c_str(), v.second.c_str());

		for (const auto& v : int_values)
			visitor.visitInt(v.first.c_str(), v.second);

		for (const auto& v : float_values)
			visitor.visitFloat(v.first.c_str(), v.second);

		for (const auto& n : nodes)
		{
			auto v = visitor.visitNode(n.first.c_str(), *n.second);
			if (v)
			{
				n.second->traverse(*v);
				v->leaveNode();
			}
		}
	}

	const char* InMemoryDatabase::fetchString(const char* key) const
	{
		auto f = string_values.find(key);

		if (f != end(string_values))
			return f->second.c_str();
		throw not_found();
	}

	bool InMemoryDatabase::fetchString(const char*& value, const char* key) const
	{
		auto f = string_values.find(key);

		if (f != end(string_values))
		{
			value = f->second.c_str();
			return true;
		}
		return false;
	}

	const char* InMemoryDatabase::queryString(const char* key, const char* default_value) const
	{
		auto f = string_values.find(key);

		if (f != end(string_values))
			return f->second.c_str();
		return default_value;
	}

	int InMemoryDatabase::fetchInt(const char* key) const
	{
		auto f = int_values.find(key);

		if (f != end(int_values))
			return f->second;
		throw not_found();
	}

	bool InMemoryDatabase::fetchInt(int& value, const char* key) const
	{
		auto f = int_values.find(key);

		if (f != end(int_values))
		{
			value = f->second;
			return true;
		}
		return false;
	}

	int InMemoryDatabase::queryInt(const char* key, int default_value) const
	{
		auto f = int_values.find(key);

		if (f != end(int_values))
			return f->second;
		return default_value;
	}

	float InMemoryDatabase::fetchFloat(const char* key) const
	{
		auto f = float_values.find(key);

		if (f != end(float_values))
			return f->second;
		throw not_found();
	}

	bool InMemoryDatabase::fetchFloat(float& value, const char* key) const
	{
		auto f = float_values.find(key);

		if (f != end(float_values))
		{
			value = f->second;
			return true;
		}
		return false;
	}

	float InMemoryDatabase::queryFloat(const char* key, float default_value) const
	{
		auto f = float_values.find(key);

		if (f != end(float_values))
			return f->second;
		return default_value;
	}


	const InMemoryDatabase& InMemoryDatabase::fetchNode(const char* key) const
	{
		auto n = nodes.find(key);

		if (n != end(nodes))
			return *n->second;
		throw not_found();
	}

	InMemoryDatabase& InMemoryDatabase::fetchNode(const char* key)
	{
		auto n = nodes.find(key);

		if (n != end(nodes))
			return *n->second;
		throw not_found();
	}

	bool InMemoryDatabase::fetchNode(const Database*& node, const char* key) const
	{
		auto n = nodes.find(key);

		if (n != end(nodes))
		{
			node = n->second.get();
			return true;
		}
		return false;
	}

	bool InMemoryDatabase::fetchNode(Database*& node, const char* key)
	{
		auto n = nodes.find(key);

		if (n != end(nodes))
		{
			node = n->second.get();
			return true;
		}
		return false;
	}

	const Database& InMemoryDatabase::queryNode(const char* key, const Database& default_node) const
	{
		auto n = nodes.find(key);

		if (n != end(nodes))
			return *n->second;
		return default_node;
	}

	Database& InMemoryDatabase::queryNode(const char* key, Database& default_node)
	{
		auto n = nodes.find(key);

		if (n != end(nodes))
			return *n->second;
		return default_node;
	}

	InMemoryDatabase& InMemoryDatabase::openNode(const char* key)
	{
		return *nodes[key];
	}


	void InMemoryDatabase::storeString(const char* key, const char* value)
	{
		string_values[key] = value;
	}

	void InMemoryDatabase::storeString(const char* key, const std::string& value)
	{
		string_values[key] = value;
	}

	void InMemoryDatabase::storeString(const char* key, std::string&& value)
	{
		string_values[key] = std::move(value);
	}

	void InMemoryDatabase::storeInt(const char* key, int value)
	{
		int_values[key] = value;
	}

	void InMemoryDatabase::storeFloat(const char* key, float value)
	{
		float_values[key] = value;
	}
}
