#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>
#include <deque>
#include <tuple>
#include <optional>

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
  
  void Clear() noexcept { CountMap_.clear(); }

 private:
  std::unordered_map<T, int, FasterStringHasher> CountMap_;
};

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

struct WobjCountPair {
  static WobjCountPair FromStringView(std::string_view sv);
  std::string_view Wobj;
  int Count;
};

inline int StringViewToInt(std::string_view sv) noexcept {
  int result = 0;
  for (const char ch : sv) {
    // skip digit check
    result = result * 10 + ch - '0';
  }
  return result;
}

WobjCountPair WobjCountPair::FromStringView(std::string_view sv) {
  const int two_spot_position = sv.find(':');
  return WobjCountPair{ sv.substr(0, two_spot_position), StringViewToInt(sv.substr(two_spot_position + 1)) };
}

template <typename TRecordVisitor>
void ProcessLines(std::istream& input, TRecordVisitor visitor) {
  std::string line;
  std::vector<std::string_view> splitted_parts;
  while (std::getline(input, line)) {
    splitted_parts.clear();
    SplitIter(line, [&splitted_parts](const auto&& part) { splitted_parts.push_back(std::move(part)); });
    visitor.NextRecord(std::string(splitted_parts.front()));
    for (size_t i = 1; i < splitted_parts.size(); ++i) {
      const WobjCountPair wobj_count = WobjCountPair::FromStringView(std::move(splitted_parts[i]));
      visitor.NextWobjCount(std::move(wobj_count));
    }
  }
}

class StringViewPool {
 public:
  std::string_view Pool(std::string_view) noexcept;
 private:
  std::unordered_set<std::string_view> StringViewsInPool_;
  std::deque<std::string> StringHolder_;
};

std::string_view StringViewPool::Pool(std::string_view sv) noexcept {
  auto pooled_sv = StringViewsInPool_.find(sv);
  if (pooled_sv == StringViewsInPool_.end()) {
    StringHolder_.emplace_back(sv);
    pooled_sv = StringViewsInPool_.emplace_hint(std::move(pooled_sv), StringHolder_.back());
  }
  return *pooled_sv;
}

class RecordCompressedPrintingVisitor {
 public:
  RecordCompressedPrintingVisitor(std::ostream& output) noexcept : Output_(output) {}
  void NextRecord(std::string date) noexcept;
  void NextWobjCount(WobjCountPair wcp) {
    Counter_.Up(SVPool_.Pool(wcp.Wobj), wcp.Count);
  }
  void PrintRecordsCompressed() const noexcept;
 private:
  std::ostream& Output_;
  int RecordsWithSameDateParsed_{0};
  StringViewPool SVPool_;
  Counter<std::string_view> Counter_;
  std::optional<std::string> LastDate_{std::nullopt};
};

void RecordCompressedPrintingVisitor::PrintRecordsCompressed() const noexcept {
  if (!Counter_.GetCountingContainer().size()) {
    return;
  }
  Output_ << *LastDate_ << " ";
  for (const auto& wobj_count_pair : Counter_.GetCountingContainer()) {
    Output_ << wobj_count_pair.first << ":" << (wobj_count_pair.second / static_cast<double>(RecordsWithSameDateParsed_)) << " ";
  }
  Output_ << "\n";
}

void RecordCompressedPrintingVisitor::NextRecord(std::string new_date) noexcept {
  if (LastDate_ && LastDate_ != new_date) {
    PrintRecordsCompressed();
    Counter_.Clear();
    RecordsWithSameDateParsed_ = 0;
  }
  LastDate_ = std::move(new_date);
  ++RecordsWithSameDateParsed_;
}

int main() {
  RecordCompressedPrintingVisitor visitor(std::cout);
  ProcessLines(std::cin, visitor);
  visitor.PrintRecordsCompressed();
  return 0;
}

