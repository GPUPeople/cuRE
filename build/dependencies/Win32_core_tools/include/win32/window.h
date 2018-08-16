


#ifndef INCLUDED_WIN32_WINDOW
#define INCLUDED_WIN32_WINDOW

#pragma once

#include "platform.h"
#include "unique_handle.h"


namespace Win32
{
	struct DestroyWindowDeleter
	{
		void operator ()(HWND hwnd) const
		{
			DestroyWindow(hwnd);
		}
	};

	using unique_hwnd = unique_handle<HWND, 0, DestroyWindowDeleter>;

	unique_hwnd createWindow(DWORD ex_style, LPCWSTR class_name, LPCWSTR window_name, DWORD style, int x, int y, int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param = nullptr);
	unique_hwnd createWindow(DWORD ex_style, LPCSTR class_name, LPCSTR window_name, DWORD style, int x, int y, int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param = nullptr);
	unique_hwnd createWindow(DWORD ex_style, ATOM class_atom, LPCWSTR window_name, DWORD style, int x, int y, int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param = nullptr);
}

#endif  // INCLUDED_WIN32_WINDOW
