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
// ************************************************************************
//@HEADER

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_

#include <mdspan/mdspan.hpp>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

template<class NestedAccessor>
class conjugated_accessor {
private:
  using nested_element_type = typename NestedAccessor::element_type;
  using nc_result_type = decltype(impl::conj_if_needed(std::declval<nested_element_type>()));
public:
  using element_type = std::add_const_t<nc_result_type>;
  using reference = std::remove_const_t<element_type>;
  using data_handle_type = typename NestedAccessor::data_handle_type;
  using offset_policy =
    conjugated_accessor<typename NestedAccessor::offset_policy>;

  constexpr conjugated_accessor() = default;
  constexpr conjugated_accessor(const NestedAccessor& acc) : nested_accessor_(acc) {}

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherNestedAccessor,
    /* requires */ (std::is_convertible_v<NestedAccessor, const OtherNestedAccessor&>)
  )
#if defined(__cpp_conditional_explicit)
  explicit(!std::is_convertible_v<OtherNestedAccessor, NestedAccessor>)
#endif
  constexpr conjugated_accessor(const conjugated_accessor<OtherNestedAccessor>& other)
    : nested_accessor_(other.nested_accessor())
  {}

  constexpr reference
    access(data_handle_type p, ::std::size_t i) const noexcept
  {
    return impl::conj_if_needed(nested_element_type(nested_accessor_.access(p, i)));
  }

  constexpr typename offset_policy::data_handle_type
    offset(data_handle_type p, ::std::size_t i) const noexcept
  {
    return nested_accessor_.offset(p, i);
  }

  const NestedAccessor& nested_accessor() const noexcept { return nested_accessor_; }

private:
  NestedAccessor nested_accessor_;
};

template<class ElementType, class Extents, class Layout, class Accessor>
auto conjugated(mdspan<ElementType, Extents, Layout, Accessor> a)
{
  using value_type = typename decltype(a)::value_type;

  // Current status of [linalg] only optimizes if Accessor is
  // conjugated_accessor<Accessor> for some Accessor.
  // There's a separate specialization for that case below.

#if defined(LINALG_FIX_CONJUGATED_FOR_NONCOMPLEX)
  // P3050 optimizes conjugated's accessor type for when
  // we know that it can't be complex: arithmetic types,
  // and types for which `conj` is not ADL-findable.
  if constexpr (std::is_arithmetic_v<value_type>) {
    return mdspan<ElementType, Extents, Layout, Accessor>
      (a.data_handle(), a.mapping(), a.accessor());
  }
  else if constexpr (! impl::has_conj<value_type>::value) {
    return mdspan<ElementType, Extents, Layout, Accessor>
      (a.data_handle(), a.mapping(), a.accessor());
  }
  else {
#endif // LINALG_FIX_CONJUGATED_FOR_NONCOMPLEX
  using return_element_type =
    typename conjugated_accessor<Accessor>::element_type;
  using return_accessor_type = conjugated_accessor<Accessor>;
  return mdspan<return_element_type, Extents, Layout, return_accessor_type>
    (a.data_handle(), a.mapping(), return_accessor_type(a.accessor()));
#if defined(LINALG_FIX_CONJUGATED_FOR_NONCOMPLEX)
  }
#endif // LINALG_FIX_CONJUGATED_FOR_NONCOMPLEX
}

// Conjugation is self-annihilating
template<class ElementType, class Extents, class Layout, class NestedAccessor>
auto conjugated(
  mdspan<ElementType, Extents, Layout, conjugated_accessor<NestedAccessor>> a)
{
  using return_element_type = typename NestedAccessor::element_type;
  using return_accessor_type = NestedAccessor;
  return mdspan<return_element_type, Extents, Layout, return_accessor_type>
    (a.data_handle(), a.mapping(), a.accessor().nested_accessor());
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CONJUGATED_HPP_
