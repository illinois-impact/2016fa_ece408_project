#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#ifdef __GNUC__
#define unused __attribute__((unused))
#else // __GNUC__
#define uinused
#endif // __GNUC__

template <typename T>
static bool check_success(const T &err);

template <>
bool check_success<herr_t>(const herr_t &err) {
  const auto res = err >= static_cast<herr_t>(0);
  assert(res);
  return res;
}

template <typename T, size_t N>
constexpr size_t array_size(const T (&)[N]) {
  return N;
}

template <typename T, typename SzTy, size_t N>
static T *zeros(const SzTy (&idims)[N]) {
  const auto dims             = std::valarray<SzTy>(idims, N);
  const auto flattened_length = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<SzTy>());
  auto res                    = new T[flattened_length];
  std::fill(res, res + N, static_cast<T>(0));
  return res;
}

#endif // __UTILS_HPP__
