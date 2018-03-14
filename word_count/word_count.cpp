#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "args/args.hxx"

struct FasterStringHasher {
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
  Counter(int minimal_frequency) : MinimalFrequncy_(minimal_frequency) {}

  void Up(const std::string& str, int up_number = 1) {
    const auto it = CountMap_.find(str);
    const int old_number = it == CountMap_.end() ? 0 : it->second;
    const int new_number = old_number + up_number;
    CountMap_.insert(std::move(it), {str, new_number});
    if (old_number < MinimalFrequncy_ && new_number >= MinimalFrequncy_) {
      FrequentWords_.push_back(str);
    }
  }

  const auto& GetFrequentWords() const noexcept { return FrequentWords_; }

  int GetCount(const std::string& str) const {
    return CountMap_.find(str)->second;
  }

 private:
  std::unordered_map<std::string, int, FasterStringHasher> CountMap_;
  std::vector<std::string> FrequentWords_;
  const int MinimalFrequncy_;
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

template <typename TCallback>
void GetFrequentWordsFromStream(std::istream& input, int frequent_barrier,
                                int skip_first_in_line,
                                TCallback frequent_words_callback) {
  std::string line;
  std::string tmp_buffer;
  Counter word_counter(frequent_barrier);
  while (std::getline(input, line)) {
    const auto splitted = Split(line);
    for (int i = skip_first_in_line; i < splitted.size(); ++i) {
      word_counter.Up(splitted[i]);
      if (i > skip_first_in_line) {
        tmp_buffer = "";
        tmp_buffer += splitted[i - 1];
        tmp_buffer += '|';
        tmp_buffer += splitted[i];
        word_counter.Up(std::move(tmp_buffer));
      }
    }
  }
  for (const auto& it : word_counter.GetFrequentWords()) {
    frequent_words_callback(it, word_counter.GetCount(it));
  }
}

bool TryFillParams(int argc, const char** argv, int* out_frequency,
                   int* out_skip_first) {
  constexpr int kDefaultMinimalWordCount = 1000;

  args::ArgumentParser parser("Count words and bigrams and save counts");
  args::HelpFlag help(parser, "help", "Display this help", {'h', "help"});
  args::ValueFlag<int> skip(parser, "skip",
                            "How many words skip in the beginning of the line",
                            {'s', "skip"}, args::Options::Required);
  args::ValueFlag<int> frequency(
      parser, "frequency",
      "Minimal frequency for a word or bigram to be printed", {'f', "freq"},
      1000);

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help&) {
    std::cout << parser;
    return false;
  } catch (const args::ParseError& e) {
    std::cerr << e.what() << "\n";
    std::cerr << parser;
    return false;
  } catch (const args::ValidationError& e) {
    std::cerr << e.what() << "\n";
    std::cerr << parser;
    return false;
  }

  *out_frequency = args::get(frequency);
  *out_skip_first = args::get(skip);
  return true;
}

int main(int argc, const char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  int minimal_frequency, skip_first;
  if (!TryFillParams(argc, argv, &minimal_frequency, &skip_first)) {
    return 1;
  }
  GetFrequentWordsFromStream(std::cin, minimal_frequency, skip_first,
                             [](const std::string& word, int count) {
                               std::cout << word << " " << count << "\n";
                             });
  return 0;
}
