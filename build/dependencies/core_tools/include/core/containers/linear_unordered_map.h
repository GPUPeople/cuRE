


#ifndef INCLUDED_CORE_CONTAINERS_LINEAR_UNORDERED_MAP
#define INCLUDED_CORE_CONTAINERS_LINEAR_UNORDERED_MAP

#pragma once

#include <type_traits>
#include <utility>
#include <tuple>
#include <vector>
#include <iterator>
#include <algorithm>


namespace core
{
	namespace containers
	{
		template <typename Key, typename Value>
		class linear_unordered_map
		{
		public:
			using key_type = Key;
			using mapped_type = Value;
			using value_type = std::pair<const Key, Value>;
			using size_type = typename std::vector<value_type>::size_type;
			using difference_type = typename std::vector<value_type>::difference_type;
			using reference = value_type&;
			using const_reference = const value_type&;

			using iterator = typename std::vector<value_type>::iterator;
			using const_iterator = typename std::vector<value_type>::const_iterator;
			using reverse_iterator = typename std::vector<value_type>::reverse_iterator;
			using const_reverse_iterator = typename std::vector<value_type>::const_reverse_iterator;

		private:
			std::vector<value_type> elements;

		public:
			auto empty() const { return elements.empty(); }
			auto size() const { return elements.size(); }
			auto max_size() const { return elements.max_size(); }

			void clear() { elements.clear(); }

			auto begin() { return elements.begin(); }
			auto begin() const { return elements.begin(); }
			auto cbegin() const { return elements.cbegin(); }

			auto end() { return elements.end(); }
			auto end() const { return elements.end(); }
			auto cend() const { return elements.cend(); }

			auto rbegin() { return elements.rbegin(); }
			auto rbegin() const { return elements.rbegin(); }
			auto rcbegin() const { return elements.rcbegin(); }

			auto rend() { return elements.rend(); }
			auto rend() const { return elements.rend(); }
			auto rcend() const { return elements.rcend(); }


			template <typename K, typename... Args>
			std::pair<iterator, bool> find_or_emplace(K&& key, Args&&... args)
			{
				auto found = find(key);
				if (found != end())
					return { found, false };
				elements.emplace_back(std::piecewise_construct, std::forward_as_tuple(std::forward<K>(key)), std::forward_as_tuple(std::forward<Args>(args)...));
				return { std::prev(elements.end()), true };
			}

			template <typename K, typename E>
			std::pair<iterator, bool> insert_or_assign(K&& key, E&& e)
			{
				auto found = find(key);
				if (found != end())
				{
					found->second = std::forward<E>(e);
					return { found, false };
				}
				elements.emplace_back(std::piecewise_construct, std::forward_as_tuple(std::forward<K>(key)), std::forward_as_tuple(std::forward<E>(e)));
				return { std::prev(elements.end()), true };
			}


			template <typename K>
			auto find(const K& key) const
			{
				return std::find_if(begin(), end(), [key](const auto& v) { return std::get<0>(v) == key; });
			}

			template <typename K>
			auto find(const K& key)
			{
				return std::find_if(begin(), end(), [key](const auto& v) { return std::get<0>(v) == key; });
			}


			template <typename K>
			mapped_type& operator[](K&& key)
			{
				return find_or_emplace(std::forward<K>(key)).first->second;
			}
		};

		using std::begin;
		using std::cbegin;
		using std::end;
		using std::cend;
	}
}

#endif  // INCLUDED_CORE_CONTAINERS_LINEAR_UNORDERED_MAP
