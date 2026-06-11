#include "pzb_line_follower_cpp/bag_reader.hpp"

#include <dirent.h>
#include <sqlite3.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace pzb {

namespace {
bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
bool is_directory(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
}  // namespace

// POSIX dirent (avoids std::filesystem, which segfaults on this Jetson toolchain).
std::string resolve_db3(const std::string& path) {
  if (!is_directory(path)) return path;
  std::string zero;
  std::vector<std::string> all;
  DIR* dir = opendir(path.c_str());
  if (dir) {
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
      std::string name = ent->d_name;
      if (ends_with(name, ".db3")) {
        std::string full = path + "/" + name;
        all.push_back(full);
        if (name.find("_0.db3") != std::string::npos) zero = full;
      }
    }
    closedir(dir);
  }
  if (!zero.empty()) return zero;
  if (!all.empty()) { std::sort(all.begin(), all.end()); return all.front(); }
  throw std::runtime_error("no .db3 found in " + path);
}

namespace {
// Run a query returning (timestamp, blob) rows for a topic, in timestamp order.
template <typename F>
void for_each_message(const std::string& db3, const std::string& topic, F&& fn) {
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(db3.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK)
    throw std::runtime_error("cannot open db3: " + db3);

  // topic id
  int topic_id = -1;
  {
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db, "SELECT id FROM topics WHERE name=?", -1, &st, nullptr);
    sqlite3_bind_text(st, 1, topic.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st) == SQLITE_ROW) topic_id = sqlite3_column_int(st, 0);
    sqlite3_finalize(st);
  }
  if (topic_id < 0) { sqlite3_close(db); return; }

  sqlite3_stmt* st = nullptr;
  sqlite3_prepare_v2(
      db, "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
      -1, &st, nullptr);
  sqlite3_bind_int(st, 1, topic_id);
  while (sqlite3_step(st) == SQLITE_ROW) {
    int64_t ts = sqlite3_column_int64(st, 0);
    const void* blob = sqlite3_column_blob(st, 1);
    int n = sqlite3_column_bytes(st, 1);
    std::vector<uint8_t> data(static_cast<const uint8_t*>(blob),
                              static_cast<const uint8_t*>(blob) + n);
    fn(ts, data);
  }
  sqlite3_finalize(st);
  sqlite3_close(db);
}

uint32_t rd_u32(const std::vector<uint8_t>& d, size_t off) {
  uint32_t v; std::memcpy(&v, d.data() + off, 4); return v;  // little-endian CDR
}
}  // namespace

cv::Mat parse_image(const std::vector<uint8_t>& data) {
  size_t off = 4;                 // encapsulation header
  off += 8;                       // stamp: int32 sec + uint32 nsec
  uint32_t flen = rd_u32(data, off); off += 4 + flen;       // frame_id string
  off = (off + 3) & ~size_t(3);   // align to 4
  uint32_t h = rd_u32(data, off); uint32_t w = rd_u32(data, off + 4); off += 8;
  uint32_t elen = rd_u32(data, off); off += 4;
  std::string enc(reinterpret_cast<const char*>(data.data() + off), elen);
  while (!enc.empty() && enc.back() == '\0') enc.pop_back();
  off += elen;
  off += 1;                       // is_bigendian
  off = (off + 3) & ~size_t(3);
  off += 4;                       // step
  off += 4;                       // pixel array length prefix
  size_t n = static_cast<size_t>(h) * w * 3;
  cv::Mat img(h, w, CV_8UC3);
  std::memcpy(img.data, data.data() + off, n);
  if (enc == "bgr8") return img;
  if (enc == "rgb8") { cv::Mat bgr; cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR); return bgr; }
  throw std::runtime_error("unsupported encoding: " + enc);
}

std::vector<ImageMsg> read_images(const std::string& db3, const std::string& topic) {
  std::vector<ImageMsg> out;
  for_each_message(db3, topic, [&](int64_t ts, const std::vector<uint8_t>& d) {
    try { out.push_back({ts, parse_image(d)}); } catch (...) {}
  });
  return out;
}

std::vector<ScalarMsg> read_int32(const std::string& db3, const std::string& topic) {
  std::vector<ScalarMsg> out;
  for_each_message(db3, topic, [&](int64_t ts, const std::vector<uint8_t>& d) {
    int32_t v; std::memcpy(&v, d.data() + 4, 4);   // 4-byte encap header, then int32
    out.push_back({ts, static_cast<double>(v)});
  });
  return out;
}

std::vector<ScalarMsg> read_float32(const std::string& db3, const std::string& topic) {
  std::vector<ScalarMsg> out;
  for_each_message(db3, topic, [&](int64_t ts, const std::vector<uint8_t>& d) {
    float v; std::memcpy(&v, d.data() + 4, 4);
    out.push_back({ts, static_cast<double>(v)});
  });
  return out;
}

std::vector<StringMsg> read_string(const std::string& db3, const std::string& topic) {
  std::vector<StringMsg> out;
  for_each_message(db3, topic, [&](int64_t ts, const std::vector<uint8_t>& d) {
    uint32_t len = rd_u32(d, 4);                   // encap header, then string length
    std::string s(reinterpret_cast<const char*>(d.data() + 8), len);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    out.push_back({ts, s});
  });
  return out;
}

}  // namespace pzb
