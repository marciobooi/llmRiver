#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace roofline {

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Minimal dependency-free JSON writer. Only supports what this tool needs:
// nested objects/arrays and scalar fields. Not a general-purpose library.
class JsonWriter {
 public:
  explicit JsonWriter(std::ostream& out) : out_(out) {}

  void begin_object() {
    before_value();
    out_ << "{";
    stack_.push_back(false);
  }
  void end_object() {
    stack_.pop_back();
    out_ << "}";
  }
  void begin_array() {
    before_value();
    out_ << "[";
    stack_.push_back(false);
  }
  void end_array() {
    stack_.pop_back();
    out_ << "]";
  }

  void key(const std::string& k) {
    comma_if_needed();
    out_ << '"' << escape(k) << "\":";
    // Suppress the comma that would otherwise be added before the value.
    just_wrote_key_ = true;
  }

  void value(const std::string& v) {
    before_value();
    out_ << '"' << escape(v) << '"';
  }
  void value(const char* v) { value(std::string(v)); }
  void value(double v) {
    before_value();
    if (std::isnan(v) || std::isinf(v)) {
      out_ << "null";
    } else {
      out_ << v;
    }
  }
  void value(int64_t v) {
    before_value();
    out_ << v;
  }
  void value(uint64_t v) {
    before_value();
    out_ << v;
  }
  void value(int v) { value(static_cast<int64_t>(v)); }
  void value(bool v) {
    before_value();
    out_ << (v ? "true" : "false");
  }
  void null_value() {
    before_value();
    out_ << "null";
  }

  void kv(const std::string& k, const std::string& v) {
    key(k);
    value(v);
  }
  void kv(const std::string& k, const char* v) { kv(k, std::string(v)); }
  void kv(const std::string& k, double v) {
    key(k);
    value(v);
  }
  void kv(const std::string& k, int64_t v) {
    key(k);
    value(v);
  }
  void kv(const std::string& k, uint64_t v) {
    key(k);
    value(v);
  }
  void kv(const std::string& k, int v) {
    key(k);
    value(v);
  }
  void kv(const std::string& k, bool v) {
    key(k);
    value(v);
  }

 private:
  void comma_if_needed() {
    if (!stack_.empty()) {
      if (stack_.back()) out_ << ",";
      stack_.back() = true;
    }
  }
  // Call before emitting any value (scalar, '{', or '['). Handles both
  // "value inside an array" (needs a comma from a sibling) and "value
  // after a key" (no comma — the key already handled separation).
  void before_value() {
    if (just_wrote_key_) {
      just_wrote_key_ = false;
      return;
    }
    comma_if_needed();
  }
  static std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      if (c == '"' || c == '\\') out += '\\';
      out += c;
    }
    return out;
  }

  std::ostream& out_;
  std::vector<bool> stack_;
  bool just_wrote_key_ = false;
};

}  // namespace roofline
