//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_SIMD_SCALAR_HPP
#define KOKKOS_SIMD_SCALAR_HPP

#include <type_traits>
#include <climits>
#include <cfloat>

#include <Kokkos_SIMD_Common.hpp>

#ifdef KOKKOS_SIMD_COMMON_MATH_HPP
#error \
    "Kokkos_SIMD_Scalar.hpp must be included before Kokkos_SIMD_Common_Math.hpp!"
#endif

namespace Kokkos {
namespace Experimental {

namespace simd_abi {

class scalar {};

}  // namespace simd_abi

template <class T>
class basic_simd_mask<T, simd_abi::scalar> {
  bool m_value;

 public:
  using value_type = bool;
  using simd_type  = basic_simd<T, simd_abi::scalar>;
  using abi_type   = simd_abi::scalar;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 1; }

  KOKKOS_DEFAULTED_FUNCTION constexpr basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(value) {}
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      basic_simd_mask<U, simd_abi::scalar> const& other) noexcept
      : m_value(static_cast<bool>(other)) {}
  template <
      class G,
      std::enable_if_t<std::is_invocable_r_v<
                           value_type, G, std::integral_constant<bool, false>>,
                       bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(gen(0)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type operator[](
      std::size_t) const {
    return m_value;
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd_mask operator!()
      const noexcept {
    return basic_simd_mask(!m_value);
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator bool()
      const noexcept {
    return m_value;
  }

  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator&&(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) && static_cast<bool>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator||(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) || static_cast<bool>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator&(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) & static_cast<bool>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator|(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) | static_cast<bool>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator^(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) ^ static_cast<bool>(rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask& operator&=(
      basic_simd_mask& lhs, basic_simd_mask const& rhs) noexcept {
    lhs &= rhs;
    return lhs;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask& operator|=(
      basic_simd_mask& lhs, basic_simd_mask const& rhs) noexcept {
    lhs |= rhs;
    return lhs;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask& operator^=(
      basic_simd_mask& lhs, basic_simd_mask const& rhs) noexcept {
    lhs ^= rhs;
    return lhs;
  }

  KOKKOS_FORCEINLINE_FUNCTION bool operator==(
      basic_simd_mask const& other) const {
    return m_value == other.m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION bool operator!=(
      basic_simd_mask const& other) const {
    return m_value != other.m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator>=(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) >= static_cast<bool>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator<=(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) <= static_cast<bool>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator>(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) > static_cast<bool>(rhs));
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd_mask operator<(
      basic_simd_mask const& lhs, basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(static_cast<bool>(lhs) < static_cast<bool>(rhs));
  }
};

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr bool all_of(
    basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar> const&
        a) noexcept {
  return static_cast<bool>(
      a == basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar>(true));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr bool any_of(
    basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar> const&
        a) noexcept {
  return static_cast<bool>(
      a != basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar>(false));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr bool none_of(
    basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar> const&
        a) noexcept {
  return static_cast<bool>(
      a == basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar>(false));
}

template <class T>
class basic_simd<T, simd_abi::scalar> {
  T m_value;

 public:
  using value_type = T;
  using abi_type   = simd_abi::scalar;
  using mask_type  = basic_simd_mask<T, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 1; }

  KOKKOS_DEFAULTED_FUNCTION constexpr basic_simd() noexcept         = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr basic_simd(basic_simd const&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr basic_simd(basic_simd&&)      = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr basic_simd& operator=(basic_simd const&) =
      default;
  KOKKOS_DEFAULTED_FUNCTION constexpr basic_simd& operator=(basic_simd&&) =
      default;
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd(U&& value) noexcept
      : m_value(value) {}
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      basic_simd<U, abi_type> const& other) noexcept
      : m_value(static_cast<U>(other)) {}
  template <class G,
            std::enable_if_t<
                // basically, can you do { value_type r =
                // gen(std::integral_constant<std::size_t, i>()); }
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
      : m_value(gen(0)) {}
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(T const* ptr,
                                                            FlagType)
      : m_value(*ptr) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator T() const {
    return m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr void copy_from(T const* ptr,
                                                       element_aligned_tag) {
    m_value = *ptr;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr void copy_from(T const* ptr,
                                                       vector_aligned_tag) {
    m_value = *ptr;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr void copy_to(
      T* ptr, element_aligned_tag) const {
    *ptr = m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr void copy_to(T* ptr,
                                                     vector_aligned_tag) const {
    *ptr = m_value;
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type operator[](
      std::size_t) const {
    return m_value;
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd operator-() const noexcept {
    return basic_simd(-m_value);
  }

  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator+(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return basic_simd(lhs.m_value + rhs.m_value);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator+(
      basic_simd const& lhs, U rhs) {
    return lhs.m_value + basic_simd(rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator-(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return basic_simd(lhs.m_value - rhs.m_value);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator-(
      basic_simd const& lhs, U rhs) {
    return lhs.m_value - basic_simd(rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator*(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return basic_simd(lhs.m_value * rhs.m_value);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator*(
      basic_simd const& lhs, U rhs) {
    return lhs.m_value * basic_simd(rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator/(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return basic_simd(lhs.m_value / rhs.m_value);
  }
  template <typename U, std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator/(
      basic_simd const& lhs, U rhs) {
    return lhs.m_value / basic_simd(rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator&(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return lhs.m_value & rhs.m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator|(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return lhs.m_value | rhs.m_value;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator<<(
      basic_simd const& lhs, int rhs) noexcept {
    return basic_simd(lhs.m_value << rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator<<(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return basic_simd(lhs.m_value << rhs.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator>>(
      basic_simd const& lhs, int rhs) noexcept {
    return basic_simd(lhs.m_value >> rhs);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator>>(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return basic_simd(lhs.m_value >> rhs.m_value);
  }

  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator+=(
      basic_simd& lhs, basic_simd const& rhs) noexcept {
    lhs = lhs + rhs;
    return lhs;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator-=(
      basic_simd& lhs, basic_simd const& rhs) noexcept {
    lhs = lhs - rhs;
    return lhs;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator*=(
      basic_simd& lhs, basic_simd const& rhs) noexcept {
    lhs = lhs * rhs;
    return lhs;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator/=(
      basic_simd& lhs, basic_simd const& rhs) noexcept {
    lhs = lhs / rhs;
    return lhs;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator<<=(
      basic_simd& lhs, basic_simd const& rhs) noexcept {
    lhs = lhs << rhs;
    return lhs;
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr basic_simd operator>>=(
      basic_simd& lhs, basic_simd const& rhs) noexcept {
    lhs = lhs >> rhs;
    return lhs;
  }

  KOKKOS_FORCEINLINE_FUNCTION friend constexpr mask_type operator==(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return mask_type(lhs.m_value == rhs.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr mask_type operator!=(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return mask_type(lhs.m_value != rhs.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr mask_type operator>=(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return mask_type(lhs.m_value >= rhs.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr mask_type operator<=(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return mask_type(lhs.m_value <= rhs.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr mask_type operator>(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return mask_type(lhs.m_value > rhs.m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION friend constexpr mask_type operator<(
      basic_simd const& lhs, basic_simd const& rhs) noexcept {
    return mask_type(lhs.m_value < rhs.m_value);
  }
};

}  // namespace Experimental

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr Experimental::basic_simd<
    T, Experimental::simd_abi::scalar>
abs(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a) {
  if constexpr (std::is_signed_v<T>) {
    return (a < 0 ? -a : a);
  }
  return a;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto floor(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a) {
  using data_type = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  return Experimental::basic_simd<data_type, Experimental::simd_abi::scalar>(
      Kokkos::floor(static_cast<data_type>(a[0])));
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto ceil(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a) {
  using data_type = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  return Experimental::basic_simd<data_type, Experimental::simd_abi::scalar>(
      Kokkos::ceil(static_cast<data_type>(a[0])));
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto round(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a) {
  using data_type = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  return Experimental::basic_simd<data_type, Experimental::simd_abi::scalar>(
      Experimental::round_half_to_nearest_even(static_cast<data_type>(a[0])));
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION constexpr auto trunc(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a) {
  using data_type = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  return Experimental::basic_simd<data_type, Experimental::simd_abi::scalar>(
      Kokkos::trunc(static_cast<data_type>(a[0])));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr Experimental::basic_simd<
    T, Experimental::simd_abi::scalar>
sqrt(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a) {
  return Experimental::basic_simd<T, Experimental::simd_abi::scalar>(
      Kokkos::sqrt(static_cast<T>(a)));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr Experimental::basic_simd<
    T, Experimental::simd_abi::scalar>
fma(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& x,
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& y,
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& z) {
  return Experimental::basic_simd<T, Experimental::simd_abi::scalar>(
      Kokkos::fma(static_cast<T>(x), static_cast<T>(y), static_cast<T>(z)));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr Experimental::basic_simd<
    T, Experimental::simd_abi::scalar>
copysign(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a,
         Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& b) {
  return Kokkos::copysign(static_cast<T>(a), static_cast<T>(b));
}

namespace Experimental {

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd<T, simd_abi::scalar> condition(
    desul::Impl::dont_deduce_this_parameter_t<
        basic_simd_mask<T, simd_abi::scalar>> const& a,
    basic_simd<T, simd_abi::scalar> const& b,
    basic_simd<T, simd_abi::scalar> const& c) {
  return basic_simd<T, simd_abi::scalar>(
      static_cast<bool>(a) ? static_cast<T>(b) : static_cast<T>(c));
}

template <class T, class BinaryOperation = std::plus<>>
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& x,
    BinaryOperation = {}) noexcept {
  return x[0];
}

template <class T, class BinaryOperation = std::plus<>>
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& x,
    Experimental::basic_simd_mask<T, Experimental::simd_abi::scalar> const&
        mask,
    BinaryOperation = {},
    T identity      = Impl::Identity<T, BinaryOperation>()) noexcept {
  if (!mask) return identity;
  return x[0];
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
template <class T, class BinaryOperation = std::plus<>>
KOKKOS_DEPRECATED_WITH_COMMENT(
    "Use reduce(basic_simd, basic_simd_mask, op, identity) instead")
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& x,
    Experimental::basic_simd_mask<T, Experimental::simd_abi::scalar> const&
        mask,
    T, BinaryOperation = {}) noexcept {
  return reduce(x, mask);
}
#endif

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce_min(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& x,
    Experimental::basic_simd_mask<T, Experimental::simd_abi::scalar> const&
        mask) noexcept {
  if (!mask) return Kokkos::reduction_identity<T>::min();
  return x[0];
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T
reduce_min(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const&
               x) noexcept {
  return x[0];
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce_max(
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& x,
    Experimental::basic_simd_mask<T, Experimental::simd_abi::scalar> const&
        mask) noexcept {
  if (!mask) return Kokkos::reduction_identity<T>::max();
  return x[0];
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T
reduce_max(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const&
               x) noexcept {
  return x[0];
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr Experimental::basic_simd<
    T, Experimental::simd_abi::scalar>
min(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a,
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& b) {
  return Experimental::basic_simd<T, Experimental::simd_abi::scalar>(
      Kokkos::min(a[0], b[0]));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr Experimental::basic_simd<
    T, Experimental::simd_abi::scalar>
max(Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& a,
    Experimental::basic_simd<T, Experimental::simd_abi::scalar> const& b) {
  return Experimental::basic_simd<T, Experimental::simd_abi::scalar>(
      Kokkos::max(a[0], b[0]));
}

template <class T>
class const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                             basic_simd<T, simd_abi::scalar>> {
 public:
  using abi_type   = simd_abi::scalar;
  using value_type = basic_simd<T, abi_type>;
  using mask_type  = basic_simd_mask<T, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  KOKKOS_FORCEINLINE_FUNCTION
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_FORCEINLINE_FUNCTION
  void copy_to(T* mem, element_aligned_tag) const {
    if (static_cast<bool>(m_mask)) *mem = static_cast<T>(m_value);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  void copy_to(T* mem, vector_aligned_tag) const {
    if (static_cast<bool>(m_mask)) *mem = static_cast<T>(m_value);
  }
  template <class Integral>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<std::is_integral_v<Integral>>
  scatter_to(T* mem,
             basic_simd<Integral, simd_abi::scalar> const& index) const {
    if (static_cast<bool>(m_mask))
      mem[static_cast<Integral>(index)] = static_cast<T>(m_value);
  }

  KOKKOS_FORCEINLINE_FUNCTION value_type const& impl_get_value() const {
    return m_value;
  }

  KOKKOS_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <class T>
class where_expression<basic_simd_mask<T, simd_abi::scalar>,
                       basic_simd<T, simd_abi::scalar>>
    : public const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                                    basic_simd<T, simd_abi::scalar>> {
  using base_type = const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                                           basic_simd<T, simd_abi::scalar>>;

 public:
  using typename base_type::value_type;
  KOKKOS_FORCEINLINE_FUNCTION
  where_expression(basic_simd_mask<T, simd_abi::scalar> const& mask_arg,
                   basic_simd<T, simd_abi::scalar>& value_arg)
      : base_type(mask_arg, value_arg) {}
  KOKKOS_FORCEINLINE_FUNCTION
  void copy_from(T const* mem, element_aligned_tag) {
    if (static_cast<bool>(this->m_mask)) this->m_value = *mem;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  void copy_from(T const* mem, vector_aligned_tag) {
    if (static_cast<bool>(this->m_mask)) this->m_value = *mem;
  }
  template <class Integral>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<std::is_integral_v<Integral>>
  gather_from(T const* mem,
              basic_simd<Integral, simd_abi::scalar> const& index) {
    if (static_cast<bool>(this->m_mask))
      this->m_value = mem[static_cast<Integral>(index)];
  }
  template <class U, std::enable_if_t<std::is_convertible_v<
                                          U, basic_simd<T, simd_abi::scalar>>,
                                      bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION void operator=(U&& x) {
    if (static_cast<bool>(this->m_mask))
      this->m_value =
          static_cast<basic_simd<T, simd_abi::scalar>>(std::forward<U>(x));
  }
};

template <class T>
KOKKOS_FORCEINLINE_FUNCTION
    where_expression<basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar>,
                     basic_simd<T, Kokkos::Experimental::simd_abi::scalar>>
    where(typename basic_simd<
              T, Kokkos::Experimental::simd_abi::scalar>::mask_type const& mask,
          basic_simd<T, Kokkos::Experimental::simd_abi::scalar>& value) {
  return where_expression(mask, value);
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION const_where_expression<
    basic_simd_mask<T, Kokkos::Experimental::simd_abi::scalar>,
    basic_simd<T, Kokkos::Experimental::simd_abi::scalar>>
where(typename basic_simd<
          T, Kokkos::Experimental::simd_abi::scalar>::mask_type const& mask,
      basic_simd<T, Kokkos::Experimental::simd_abi::scalar> const& value) {
  return const_where_expression(mask, value);
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
template <class T>
KOKKOS_DEPRECATED_WITH_COMMENT("Use reduce_max() instead")
KOKKOS_FORCEINLINE_FUNCTION T
    hmax(const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                                basic_simd<T, simd_abi::scalar>> const& x) {
  return static_cast<bool>(x.impl_get_mask())
             ? static_cast<T>(x.impl_get_value())
             : Kokkos::reduction_identity<T>::max();
}
#endif

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce(
    const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                           basic_simd<T, simd_abi::scalar>> const& x,
    T identity_element, std::plus<>) {
  return static_cast<bool>(x.impl_get_mask())
             ? static_cast<T>(x.impl_get_value())
             : identity_element;
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce_max(
    const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                           basic_simd<T, simd_abi::scalar>> const& x) noexcept {
  return static_cast<bool>(x.impl_get_mask())
             ? static_cast<T>(x.impl_get_value())
             : Kokkos::reduction_identity<T>::max();
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
template <class T>
KOKKOS_DEPRECATED_WITH_COMMENT("Use reduce_min() instead")
KOKKOS_FORCEINLINE_FUNCTION T
    hmin(const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                                basic_simd<T, simd_abi::scalar>> const& x) {
  return static_cast<bool>(x.impl_get_mask())
             ? static_cast<T>(x.impl_get_value())
             : Kokkos::reduction_identity<T>::min();
}
#endif

template <class T>
KOKKOS_FORCEINLINE_FUNCTION constexpr T reduce_min(
    const_where_expression<basic_simd_mask<T, simd_abi::scalar>,
                           basic_simd<T, simd_abi::scalar>> const& x) noexcept {
  return static_cast<bool>(x.impl_get_mask())
             ? static_cast<T>(x.impl_get_value())
             : Kokkos::reduction_identity<T>::min();
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
