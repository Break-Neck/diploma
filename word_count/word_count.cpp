#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int kWordsStartPos = 2;
constexpr int kMinemalWordCount = 1000;

struct FastStringHasher {
 public:
  constexpr static size_t kMultiplier = 17;

  size_t operator()(const std::string& str) const noexcept {
    size_t hash = 0;
    for (int i = 0; i < str.length(); ++i) {
      hash = hash * kMultiplier + str[i];
    }
    return hash;
  }
};

class Counter {
 public:
  void Up(const std::string& str) {
    const auto it = count_map.find(str);
    if (it == count_map.end()) {
      count_map.insert({str, 1});
    } else {
      if (++(it->second) == kMinemalWordCount) {
        frequent_words.push_back(str);
      }
    }
  }

  const auto& GetFrequentWords() const noexcept { return frequent_words; }

  int GetCount(const std::string& str) const {
    return count_map.find(str)->second;
  }

 private:
  std::unordered_map<std::string, int, FastStringHasher> count_map;
  std::vector<std::string> frequent_words;
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
  Counter word_counter;
  while (std::getline(std::cin, line)) {
    const auto splitted = Split(line);
    for (int i = kWordsStartPos; i < splitted.size(); ++i) {
      word_counter.Up(splitted[i]);
      if (i > kWordsStartPos) {
        word_counter.Up(splitted[i - 1] + '|' + splitted[i]);
      }
    }
  }
  for (const auto& it : word_counter.GetFrequentWords()) {
    std::cout << it << " " << word_counter.GetCount(it) << "\n";
  }
  return 0;
}
