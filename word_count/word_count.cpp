#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "args/args.hxx"

struct FasterStringHasher {
 public:
  constexpr static size_t kMultiplier = 17;

  size_t operator()(const std::string& str) const noexcept {
    size_t hash = 0;
    for (size_t i = 0; i < str.length(); ++i) {
      hash = hash * kMultiplier + str[i];
    }
    return hash;
  }
};

class StringCounter {
 public:
  int Up(const std::string& str, int up_number = 1) {
    const auto it = CountMap_.find(str);
    const int old_number = it == CountMap_.end() ? 0 : it->second;
    const int new_number = old_number + up_number;
    CountMap_.insert_or_assign(std::move(it), str, new_number);
    return old_number;
  }

  int GetCount(const std::string& str) const {
    const auto it = CountMap_.find(str);
    return it != CountMap_.end() ? it->second : 0;
  }

 private:
  std::unordered_map<std::string, int, FasterStringHasher> CountMap_;
};

class StringCounterWithFrequncyFiltering : public StringCounter {
 public:
  StringCounterWithFrequncyFiltering(int minimal_frequency)
      : MinimalFrequncy_(minimal_frequency) {}

  int Up(const std::string& str, int up_number = 1) {
    const int old_number = StringCounter::Up(str, up_number);
    const int new_number = old_number + up_number;
    if (old_number < MinimalFrequncy_ && new_number >= MinimalFrequncy_) {
      FrequentWords_.push_back(str);
    }
    return old_number;
  }

  const auto& GetFrequentWords() const { return FrequentWords_; }

 private:
  std::vector<std::string> FrequentWords_;
  const int MinimalFrequncy_;
};

std::vector<std::string> Split(const std::string& str) {
  size_t start = 0;
  std::vector<std::string> result;
  for (size_t i = 0; i < str.length(); ++i) {
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

template <typename TVisitor>
void ProcessWordsFromStream(std::istream& input, int skip_first_in_line,
                            TVisitor visitor) {
  std::string line;
  while (std::getline(input, line)) {
    const auto splitted = Split(line);
    for (size_t i = skip_first_in_line; i < splitted.size(); ++i) {
      visitor.OnWordObject(splitted[i]);
      if (i > static_cast<size_t>(skip_first_in_line)) {
        visitor.OnWordObject(splitted[i - 1] + '|' + splitted[i]);
      }
    }
    visitor.OnNewDocument();
  }
}

bool TryFillParams(int argc, const char** argv, int* out_frequency,
                   int* out_skip_first) {
  constexpr int kDefaultMinimalWordCount = 1000;

  args::ArgumentParser parser(
      "Count words and bigrams and save counts for words and number of "
      "documents containing this word");
  args::HelpFlag help(parser, "help", "Display this help", {'h', "help"});
  args::ValueFlag<int> skip(parser, "skip",
                            "How many words skip in the beginning of the line",
                            {'s', "skip"}, args::Options::Required);
  args::ValueFlag<int> frequency(
      parser, "frequency",
      "Minimal frequency for a word or bigram to be printed", {'f', "freq"},
      kDefaultMinimalWordCount);

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

template <typename TWordCounter, typename TDocumentContainingWordCounter,
          typename TDocumentWordsSet>
struct CountingVisitor {
  TWordCounter& WordCounter;
  TDocumentContainingWordCounter& DocumentContainingWordCounter;
  TDocumentWordsSet& DocumentWordsSet;

  CountingVisitor(
      TWordCounter& word_counter,
      TDocumentContainingWordCounter& document_containig_word_counter,
      TDocumentWordsSet& document_words_set)
      : WordCounter(word_counter),
        DocumentContainingWordCounter(document_containig_word_counter),
        DocumentWordsSet(document_words_set) {}

  void OnNewDocument() {
    for (const auto& word : DocumentWordsSet) {
      DocumentContainingWordCounter.Up(word);
    }
    DocumentWordsSet.clear();
  }

  void OnWordObject(const std::string& word) {
    WordCounter.Up(word);
    DocumentWordsSet.insert(word);
  }
};

int main(int argc, const char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  int minimal_frequency, skip_first;
  if (!TryFillParams(argc, argv, &minimal_frequency, &skip_first)) {
    return 1;
  }
  StringCounterWithFrequncyFiltering frequent_words_counter(minimal_frequency);
  StringCounter documents_containing_word_counter;
  std::unordered_set<std::string, FasterStringHasher> current_document_words;
  CountingVisitor counting_visitor(frequent_words_counter,
                                   documents_containing_word_counter,
                                   current_document_words);

  ProcessWordsFromStream(std::cin, skip_first, std::move(counting_visitor));

  for (const auto& word : frequent_words_counter.GetFrequentWords()) {
    std::cout << word << " " << frequent_words_counter.GetCount(word) << " "
              << documents_containing_word_counter.GetCount(word) << "\n";
  }
  return 0;
}
