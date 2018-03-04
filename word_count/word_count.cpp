#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

constexpr int kWordsStartPos = 2;
constexpr int kMinemalWordCount = 1000;

template <typename T>
class Counter {
 public:
  void Up(const std::string& str) {
    const auto it = count_map.find(str);
    if (it == count_map.end()) {
      count_map.insert({str, 1});
    } else {
      ++(it->second);
    }
  }

  const auto& GetMap() const noexcept { return count_map; }

 private:
  std::unordered_map<std::string, int> count_map;
};

std::vector<std::string> Split(const std::string& str) {
  int start = 0;
  std::vector<std::string> result;
  for (int i = 0; i < str.length(); ++i) {
    if (std::isspace(str[i]) && start < i) {
      result.push_back(str.substr(start, i - start));
      start = i + 1;
    }
  }
  if (start < str.length()) {
    result.push_back(str.substr(start, str.length() - start));
  }
  return result;
}

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::string line;
  Counter<std::string> word_counter;
  while (std::getline(std::cin, line)) {
    const auto splitted = Split(line);
    for (int i = kWordsStartPos; i < splitted.size(); ++i) {
      word_counter.Up(splitted[i]);
      if (i > kWordsStartPos) {
        word_counter.Up(splitted[i - 1] + '|' + splitted[i]);
      }
    }
  }
  for (const auto& it : word_counter.GetMap()) {
    if (it.second >= kMinemalWordCount) {
      std::cout << it.first << " " << it.second << "\n";
    }
  }
  return 0;
}
