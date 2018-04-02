#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

template <typename T>
class Counter {
 public:
  int Up(const T& obj, int up_number = 1) {
    const auto it = CountMap_.find(obj);
    const int old_number = it == CountMap_.end() ? 0 : it->second;
    const int new_number = old_number + up_number;
    CountMap_.insert_or_assign(std::move(it), obj, new_number);
    return old_number;
  }

  int GetCount(const T& obj) const {
    const auto it = CountMap_.find(obj);
    return it != CountMap_.end() ? it->second : 0;
  }

  const auto& GetCountingContainer() const noexcept { return CountMap_; }

 private:
  std::unordered_map<T, int, FasterStringHasher> CountMap_;
};

auto GetGoodWords(std::istream& input) {
  std::vector<std::string> good_words;
  std::string line;
  while (std::getline(input, line)) {
    good_words.emplace_back(line.substr(0, line.find(' ')));
  }
  return good_words;
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

template <typename TWordsSet>
auto GetCountsOfWObjs(std::string_view str, const TWordsSet& good_words) {
  Counter<std::string_view> wobj_count;
  std::optional<std::string_view> last_word = std::nullopt;
  std::string temp_buffer;
  temp_buffer.reserve(100);
  SplitIter(str, [&wobj_count, &good_words, &last_word,
                  &temp_buffer](std::string_view word) {
    if (good_words.count(word)) {
      wobj_count.Up(word);
      if (last_word && good_words.count(*last_word)) {
        temp_buffer = *last_word;
        temp_buffer += "|";
        temp_buffer += word;
        const auto it = good_words.find({temp_buffer});
        if (it != good_words.end()) {
          wobj_count.Up(*it);
        }
      }
    }
    last_word = std::move(word);
  });
  return wobj_count;
}

template <typename TWordsSet, typename TCallback>
void ProcessLemmasLines(std::istream& input, const TWordsSet& good_words,
                        TCallback callback) {
  std::string line;
  while (std::getline(input, line)) {
    const int lemmas_start_in_string = line.find(' ') + 1;
    const auto words_count =
        GetCountsOfWObjs({line.data() + lemmas_start_in_string}, good_words);
    callback(std::string_view(line.data(), lemmas_start_in_string - 1),
             words_count);
  }
}

int main(int argc, char** argv) {
  using namespace std::literals;
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  if (argc != 2 || argv[1] == "-h"s) {
    std::cout
        << "Usage: count_compress good_words_file <coursed_lemmas_file \n";
    return 1;
  }
  std::ifstream good_words_input(argv[1]);
  const auto good_words = GetGoodWords(good_words_input);
  good_words_input.close();
  const std::unordered_set<std::string_view, FasterStringHasher> good_words_set(
      good_words.begin(), good_words.end());
  ProcessLemmasLines(
      std::cin, good_words_set,
      [](std::string_view date_string_view, const auto& words_count) {
        std::cout << date_string_view << " ";
        for (const auto& word_count_pair : words_count.GetCountingContainer()) {
          std::cout << word_count_pair.first << ":" << word_count_pair.second
                    << " ";
        }
        std::cout << "\n";
      });
  return 0;
}
