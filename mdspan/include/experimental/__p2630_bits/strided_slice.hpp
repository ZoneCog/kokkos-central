
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

#pragma once

#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

namespace detail {
  template<class T>
  struct mdspan_is_integral_constant: std::false_type {};

  template<class T, T val>
  struct mdspan_is_integral_constant<std::integral_constant<T,val>>: std::true_type {};
}

// Slice Specifier allowing for strides and compile time extent
template <class OffsetType, class ExtentType, class StrideType>
struct strided_slice {
  using offset_type = OffsetType;
  using extent_type = ExtentType;
  using stride_type = StrideType;

  MDSPAN_IMPL_NO_UNIQUE_ADDRESS OffsetType offset{};
  MDSPAN_IMPL_NO_UNIQUE_ADDRESS ExtentType extent{};
  MDSPAN_IMPL_NO_UNIQUE_ADDRESS StrideType stride{};

  static_assert(std::is_integral_v<OffsetType> || detail::mdspan_is_integral_constant<OffsetType>::value);
  static_assert(std::is_integral_v<ExtentType> || detail::mdspan_is_integral_constant<ExtentType>::value);
  static_assert(std::is_integral_v<StrideType> || detail::mdspan_is_integral_constant<StrideType>::value);
};

} // MDSPAN_IMPL_STANDARD_NAMESPACE
