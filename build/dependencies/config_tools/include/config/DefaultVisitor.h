


#ifndef INCLUDED_CONFIG_DEFAULT_VISITOR
#define INCLUDED_CONFIG_DEFAULT_VISITOR

#pragma once

#include "Database.h"


namespace config
{
	class DefaultVisitor : public virtual Visitor
	{
	public:
		void visitString(const char* key, const char* value) override;
		void visitInt(const char* key, int value) override;
		void visitFloat(const char* key, float value) override;
		Visitor* visitNode(const char* key, const Database& node) override;
		void leaveNode() override;
	};
}

#endif  // INCLUDED_CONFIG_DEFAULT_VISITOR
