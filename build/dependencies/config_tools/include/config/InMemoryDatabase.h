


#ifndef INCLUDED_CONFIG_INMEMORYDATABASE
#define INCLUDED_CONFIG_INMEMORYDATABASE

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "Database.h"


namespace config
{
	class InMemoryDatabase : public virtual Database
	{
		std::unordered_map<std::string, std::string> string_values;
		std::unordered_map<std::string, int> int_values;
		std::unordered_map<std::string, float> float_values;

		class ChildNode
		{
			std::unique_ptr<InMemoryDatabase> node;

		public:
			ChildNode();

			const InMemoryDatabase* get() const { return node.get(); }
			InMemoryDatabase* get() { return node.get(); }

			const InMemoryDatabase& operator *() const { return *node; }
			InMemoryDatabase& operator *() { return *node; }

			const InMemoryDatabase* operator ->() const { return node.get(); }
			InMemoryDatabase* operator ->() { return node.get(); }
		};

		std::unordered_map<std::string, ChildNode> nodes;

	public:
		void traverse(Visitor& visitor) const override;

		const char* fetchString(const char* key) const override;
		bool fetchString(const char*& value, const char* key) const override;
		const char* queryString(const char* key, const char* default_value) const override;

		int fetchInt(const char* key) const override;
		bool fetchInt(int& value, const char* key) const override;
		int queryInt(const char* key, int default_value) const override;

		float fetchFloat(const char* key) const override;
		bool fetchFloat(float& value, const char* key) const override;
		float queryFloat(const char* key, float default_value) const override;

		const InMemoryDatabase& fetchNode(const char* key) const override;
		InMemoryDatabase& fetchNode(const char* key) override;
		bool fetchNode(const Database*& node, const char* key) const override;
		bool fetchNode(Database*& node, const char* key) override;
		const Database& queryNode(const char* key, const Database& default_node = null_node) const override;
		Database& queryNode(const char* key, Database& default_node) override;
		InMemoryDatabase& openNode(const char* key) override;

		void storeString(const char* key, const char* value) override;
		void storeString(const char* key, const std::string& value) override;
		void storeString(const char* key, std::string&& value) override;
		void storeInt(const char* key, int value) override;
		void storeFloat(const char* key, float value) override;
	};
}

#endif  // INCLUDED_CONFIG_INMEMORYDATABASE
