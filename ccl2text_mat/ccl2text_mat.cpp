#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

std::vector<std::string> LoadAllWords(std::istream& input) {
  std::string line;
  std::vector<std::string> result;
  while (std::getline(input, line)) {
    result.push_back(line.substr(0, line.find(' ')));
  }
  return result;
}

struct FasterStringHasher {
 public:
  constexpr static size_t kMultiplier = 17;

  template <typename TStringType>
  size_t operator()(const TStringType& str) const noexcept {
    size_t hash = 0;
    for (size_t i = 0; i < str.length(); ++i) {
      hash = hash * kMultiplier + str[i];
    }
    return hash;
  }
};

using SVHashTable =
    std::unordered_map<std::string_view, int, FasterStringHasher>;

SVHashTable GetMap(const std::vector<std::string>& words) {
  SVHashTable table;
  table.reserve(words.size());
  for (size_t i = 0; i < words.size(); ++i) {
    table[{words[i]}] = i;
  }
  return table;
}

template <typename TCallback>
void SplitIter(std::string_view str, TCallback callback) {
  size_t start = 0;
  for (size_t i = 0; i < str.length(); ++i) {
    if (std::isspace(str[i]) && start < i) {
      callback(std::string_view(str.data() + start, i - start));
      start = i + 1;
    }
  }
  if (start < str.length()) {
    callback(std::string_view(str.data() + start, str.length() - start));
  }
}

inline int StringViewToInt(std::string_view sv) noexcept {
  int result = 0;
  for (const char ch : sv) {
    // skip digit check
    result = result * 10 + ch - '0';
  }
  return result;
}

void PrintAllElements(std::istream& input, std::ostream& output,
                      const SVHashTable& index_table) {
  std::string line;
  int line_index = 0;
  while (std::getline(input, line)) {
    SplitIter(
        line, [&index_table, line_index](std::string_view wobj_with_count) {
          {
            const auto separator_position = wobj_with_count.find(':');
            if (separator_position == std::string_view::npos) {
              return;
            }
            const std::string_view wobj =
                wobj_with_count.substr(0, separator_position);
            const int count =
                StringViewToInt(wobj_with_count.substr(separator_position + 1));
            std::cout << line_index << " " << index_table.find(wobj)->second << " "
                      << count << "\n";
          }
        });
    ++line_index;
  }
}

int main(int argc, char** argv) {
  using namespace std::literals;
  if (argc != 2 || argv[1] == "-h"s || argv[1] == "--help"s) {
    std::cout << "One parameter is needed: path to good words file. \n";
    return 1;
  }
  std::ifstream words_file(argv[1]);
  const auto all_words = LoadAllWords(words_file);
  words_file.close();
  const auto words_to_index = GetMap(all_words);
  PrintAllElements(std::cin, std::cout, words_to_index);
  return 0;
}
