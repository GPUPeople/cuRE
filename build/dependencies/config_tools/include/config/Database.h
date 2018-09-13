


#ifndef INCLUDED_CONFIG_DATABASE
#define INCLUDED_CONFIG_DATABASE

#pragma once

#include <string>

#include <core/interface>


namespace config
{
	struct Database;

	struct INTERFACE Visitor
	{
		virtual void visitString(const char* key, const char* value) = 0;
		virtual void visitInt(const char* key, int value) = 0;
		virtual void visitFloat(const char* key, float value) = 0;
		virtual Visitor* visitNode(const char* key, const Database& node) = 0;
		virtual void leaveNode() = 0;

	protected:
		Visitor() = default;
		Visitor(Visitor&&) = default;
		Visitor(const Visitor&) = default;
		Visitor& operator =(Visitor&&) = default;
		Visitor& operator =(const Visitor&) = default;
		~Visitor() = default;
	};

	struct INTERFACE Database
	{
		static const Database& null_node;

		virtual void traverse(Visitor& visitor) const = 0;

		virtual const char* fetchString(const char* key) const = 0;
		virtual bool fetchString(const char*& value, const char* key) const = 0;
		virtual const char* queryString(const char* key, const char* default_value) const = 0;

		virtual int fetchInt(const char* key) const = 0;
		virtual bool fetchInt(int& value, const char* key) const = 0;
		virtual int queryInt(const char* key, int default_value) const = 0;

		virtual float fetchFloat(const char* key) const = 0;
		virtual bool fetchFloat(float& value, const char* key) const = 0;
		virtual float queryFloat(const char* key, float default_value) const = 0;

		virtual const Database& fetchNode(const char* key) const = 0;
		virtual Database& fetchNode(const char* key) = 0;
		virtual bool fetchNode(const Database*& node, const char* key) const = 0;
		virtual bool fetchNode(Database*& node, const char* key) = 0;
		virtual const Database& queryNode(const char* key, const Database& default_node = null_node) const = 0;
		virtual Database& queryNode(const char* key, Database& default_node) = 0;
		virtual Database& openNode(const char* key) = 0;

		virtual void storeString(const char* key, const char* value) = 0;
		virtual void storeString(const char* key, const std::string& value) = 0;
		virtual void storeString(const char* key, std::string&& value) = 0;
		virtual void storeInt(const char* key, int value) = 0;
		virtual void storeFloat(const char* key, float value) = 0;

	protected:
		Database() = default;
		Database(Database&&) = default;
		Database(const Database&) = default;
		Database& operator=(Database&&) = default;
		Database& operator=(const Database&) = default;
		~Database() = default;
	};
}

#endif  // INCLUDED_CONFIG_DATABASE
