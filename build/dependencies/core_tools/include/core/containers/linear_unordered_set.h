


#ifndef INCLUDED_CORE_CONTAINERS_LINEAR_UNORDERED_SET
#define INCLUDED_CORE_CONTAINERS_LINEAR_UNORDERED_SET

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
		template <typename Key>
		class linear_unordered_set
		{
		public:
			using key_type = Key;
			using value_type = Key;
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


			template <typename K>
			auto find(const K& key) const
			{
				return std::find(begin(), end(), key);
			}

			template <typename K>
			auto find(const K& key)
			{
				return std::find(begin(), end(), key);
			}


			template <typename K>
			std::pair<iterator, bool> insert(const K& key)
			{
				if (auto it = find(key); it != end())
					return { it, false };
				return { elements.insert(end(), key), false };
			}

			template <typename K>
			std::pair<iterator, bool> insert(K&& key)
			{
				if (auto it = find(key); it != end())
					return { it, false };
				return { elements.insert(end(), key), false };
			}
		};

		using std::begin;
		using std::cbegin;
		using std::end;
		using std::cend;
	}
}

#endif  // INCLUDED_CORE_CONTAINERS_LINEAR_UNORDERED_SET
