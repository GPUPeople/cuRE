


#ifndef INCLUDED_CONFIG_OVERLAY
#define INCLUDED_CONFIG_OVERLAY

#pragma once

#include <initializer_list>
#include <vector>

#include "Database.h"


namespace config
{
	class DatabaseOverlay : public virtual Database
	{
		std::vector<const Database*> overlays;
		Database& base;

	public:
		DatabaseOverlay(Database& base, std::initializer_list<const Database*> overlays);

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

		const Database& fetchNode(const char* key) const override;
		Database& fetchNode(const char* key) override;
		bool fetchNode(const Database*& node, const char* key) const override;
		bool fetchNode(Database*& node, const char* key) override;
		const Database& queryNode(const char* key, const Database& default_node = null_node) const override;
		Database& queryNode(const char* key, Database& default_node) override;
		Database& openNode(const char* key) override;

		void storeString(const char* key, const char* value) override;
		void storeString(const char* key, const std::string& value) override;
		void storeString(const char* key, std::string&& value) override;
		void storeInt(const char* key, int value) override;
		void storeFloat(const char* key, float value) override;
	};
}

#endif  // INCLUDED_CONFIG_OVERLAY
