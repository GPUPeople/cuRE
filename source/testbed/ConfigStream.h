


#ifndef INCLUDED_CONFIGFILE_STREAM
#define INCLUDED_CONFIGFILE_STREAM

#pragma once

#include <stdexcept>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "interface.h"


namespace ConfigFile
{
	class lexer_error : public std::runtime_error
	{
	public:
		lexer_error(const std::string& msg)
			: runtime_error(msg)
		{
		}
	};

	struct Token
	{
		const char* begin;
		const char* end;

		Token(const char* begin = nullptr)
			: begin(begin), end(begin)
		{
		}

		Token(const char* begin, const char* end)
			: begin(begin), end(end)
		{
		}

		Token(const char* begin, size_t length)
			: begin(begin), end(begin + length)
		{
		}
	};

	inline bool operator ==(const Token& a, const Token& b)
	{
		if (a.end - a.begin != b.end - b.begin)
			return false;
		for (auto c1 = a.begin, c2 = b.begin; c1 != a.end; ++c1, ++c2)
			if (*c1 != *c2)
				return false;
		return true;
	}

	inline bool operator ==(const Token& t, const char* str)
	{
		return std::strncmp(t.begin, str, t.end - t.begin) == 0;
	}

	inline bool operator ==(const char* str, const Token& t)
	{
		return t == str;
	}

	inline bool operator !=(const Token& a, const Token& b)
	{
		return !(a == b);
	}

	inline bool operator !=(const Token& t, const char* str)
	{
		return std::strncmp(t.begin, str, t.end - t.begin) != 0;
	}

	inline bool operator !=(const char* str, const Token& t)
	{
		return t != str;
	}

	inline bool empty(const Token& t)
	{
		return t.begin == t.end;
	}


	enum class OPERATOR_TOKEN
	{
		INVALID = -1,
		PLUS,
		MINUS,
		ASTERISK,
		SLASH,
		CIRCUMFLEX,
		TILDE,
		LPARENT,
		RPARENT,
		LBRACKET,
		RBRACKET,
		LBRACE,
		RBRACE,
		QUEST,
		DOT,
		COLON,
		COMMA,
		SEMICOLON,
		EQ,
		EEQ,
		NEQ,
		LT,
		LEQ,
		LL,
		GT,
		GEQ,
		GG,
		AND,
		AAND,
		OR,
		OOR,
		BANG,
		PERCENT,
		ARROW
	};

	Token token(OPERATOR_TOKEN op);


	class Stream;

	class INTERFACE LexerCallback
	{
	protected:
		LexerCallback() = default;
		LexerCallback(const LexerCallback&) = default;
		~LexerCallback() = default;
		LexerCallback& operator =(const LexerCallback&) = default;
	public:
		virtual bool consumeComment(Stream& stream, Token t) = 0;
		virtual bool consumeIdentifier(Stream& stream, Token t) = 0;
		virtual bool consumeIntegerLiteral(Stream& stream, Token t) = 0;
		virtual bool consumeFloatLiteral(Stream& stream, Token t) = 0;
		virtual bool consumeStringLiteral(Stream& stream, Token t) = 0;
		virtual bool consumeOperator(Stream& stream, OPERATOR_TOKEN op, Token t) = 0;
		virtual bool consumeEOL(Stream& stream) = 0;
		virtual void consumeEOF(Stream& stream) = 0;
	};

	class INTERFACE Log
	{
	protected:
		Log() {}
		Log(const Log&) {}
		~Log() {}
		Log& operator =(const Log&) { return *this; }
	public:
		virtual void warning(const char* message, const char* file, size_t line, ptrdiff_t column) = 0;
		virtual void warning(const std::string& message, const char* file, size_t line, ptrdiff_t column) = 0;
		virtual void error(const char* message, const char* file, size_t line, ptrdiff_t column) = 0;
		virtual void error(const std::string& message, const char* file, size_t line, ptrdiff_t column) = 0;
	};


	class Stream
	{
	private:
		const char* ptr;
		const char* end;

		std::vector<const char*> lines;

		const char* stream_name;

		Log& log;

		size_t getCurrentLineNumber();
		ptrdiff_t getCurrentColumn();

	public:
		Stream(const Stream&) = delete;
		Stream& operator =(const Stream&) = delete;

		Stream(const char* begin, const char* end, const char* name, Log& log);

		bool eof() const { return ptr >= end; }

		const char* current() const
		{
			return ptr;
		}

		const char* get()
		{
			if (*ptr == '\n')
			{
				lines.push_back(ptr + 1);
			}
			return ptr++;
		}

		void warning(const char* message);
		void warning(const std::string& message);
		void error(const char* message);
		void error(const std::string& message);
	};

	Stream& consume(Stream& stream, LexerCallback& callback);
}

#endif  // INCLUDED_CONFIG_STREAM
