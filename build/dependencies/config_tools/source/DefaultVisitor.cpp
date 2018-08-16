


#include <config/DefaultVisitor.h>


namespace config
{
	void DefaultVisitor::visitString(const char* key, const char* value)
	{
	}

	void DefaultVisitor::visitInt(const char* key, int value)
	{
	}

	void DefaultVisitor::visitFloat(const char* key, float value)
	{
	}

	Visitor* DefaultVisitor::visitNode(const char* key, const Database& node)
	{
		return this;
	}

	void DefaultVisitor::leaveNode()
	{
	}
}
