


#include <config/error.h>
#include <config/Database.h>


namespace
{
	class NullDatabase : public virtual config::Database
	{
	public:
		void traverse(config::Visitor& visitor) const override {}

		const char* fetchString(const char* key) const override
		{
			throw config::not_found();
		}
		bool fetchString(const char*& value, const char* key) const override
		{
			return false;
		}
		const char* queryString(const char* key, const char* default_value) const override
		{
			return default_value;
		}

		int fetchInt(const char* key) const override
		{
			throw config::not_found();
		}
		bool fetchInt(int& value, const char* key) const override
		{
			return false;
		}
		int queryInt(const char* key, int default_value) const override
		{
			return default_value;
		}

		float fetchFloat(const char* key) const override
		{
			throw config::not_found();
		}
		bool fetchFloat(float& value, const char* key) const override
		{
			return false;
		}
		float queryFloat(const char* key, float default_value) const override
		{
			return default_value;
		}

		const Database& fetchNode(const char* key) const override
		{
			throw config::not_found();
		}
		Database& fetchNode(const char* key) override
		{
			throw config::not_found();
		}
		bool fetchNode(const Database*& node, const char* key) const override
		{
			return false;
		}
		bool fetchNode(Database*& node, const char* key) override
		{
			return false;
		}
		const Database& queryNode(const char* key, const Database& default_node) const override
		{
			return default_node;
		}
		Database& queryNode(const char* key, Database& default_node) override
		{
			return default_node;
		}
		Database& openNode(const char* key) override
		{
			throw config::not_found();
		}

		void storeString(const char* key, const char* value) override {}
		void storeString(const char* key, const std::string& value) override {}
		void storeString(const char* key, std::string&& value) override {}
		void storeInt(const char* key, int value) override {}
		void storeFloat(const char* key, float value) override {}
	} null_database;
}

namespace config
{
	const Database& Database::null_node = null_database;
}
