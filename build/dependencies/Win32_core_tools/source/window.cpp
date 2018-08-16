


#include <win32/error.h>

#include <win32/window.h>


namespace Win32
{
	unique_hwnd createWindow(DWORD ex_style, LPCWSTR class_name, LPCWSTR window_name, DWORD style, int x, int y, int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param)
	{
		auto hwnd = CreateWindowExW(ex_style, class_name, window_name, style, x, y, width, height, parent, menu, instance, param);
		if (hwnd == 0)
			throw_last_error();
		return unique_hwnd { hwnd };
	}

	unique_hwnd createWindow(DWORD ex_style, LPCSTR class_name, LPCSTR window_name, DWORD style, int x, int y, int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param)
	{
		auto hwnd = CreateWindowExA(ex_style, class_name, window_name, style, x, y, width, height, parent, menu, instance, param);
		if (hwnd == 0)
			throw_last_error();
		return unique_hwnd { hwnd };
	}

	unique_hwnd createWindow(DWORD ex_style, ATOM class_atom, LPCWSTR window_name, DWORD style, int x, int y, int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param)
	{
		return createWindow(ex_style, reinterpret_cast<LPCWSTR>(class_atom), window_name, style, x, y, width, height, parent, menu, instance, param);
	}
}
