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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_HPP_

#include <complex>
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace MDSPAN_IMPL_PROPOSED_NAMESPACE {
inline namespace __p1673_version_0 {
namespace linalg {

namespace {

// Overwriting general matrix-vector product: y := A * x
template <class Exec, class A_t, class X_t, class Y_t, class = void>
struct is_custom_mat_vec_product_avail : std::false_type {};

template <class Exec, class A_t, class X_t, class Y_t>
struct is_custom_mat_vec_product_avail<
  Exec, A_t, X_t, Y_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(
	       matrix_vector_product
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<X_t>(),
		std::declval<Y_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Overwriting general matrix-vector product with update: z := y + A * x
template <class Exec, class A_t, class X_t, class Y_t, class Z_t, class = void>
struct is_custom_mat_vec_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class X_t, class Y_t, class Z_t>
struct is_custom_mat_vec_product_with_update_avail<
  Exec, A_t, X_t, Y_t, Z_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(matrix_vector_product(std::declval<Exec>(),
				     std::declval<A_t>(),
				     std::declval<X_t>(),
				     std::declval<Y_t>(),
				     std::declval<Z_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Overwriting symmetric matrix-vector product
template <class Exec, class A_t, class Triangle, class X_t, class Y_t, class = void>
struct is_custom_sym_mat_vec_product_avail : std::false_type {};

template <class Exec, class A_t, class Triangle, class X_t, class Y_t>
struct is_custom_sym_mat_vec_product_avail<
  Exec, A_t, Triangle, X_t, Y_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_vector_product(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<Triangle>(),
					       std::declval<X_t>(),
					       std::declval<Y_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Overwriting symmetric matrix-vector product with update
template <class Exec, class A_t, class Triangle, class X_t, class Y_t, class Z_t, class = void>
struct is_custom_sym_mat_vec_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Triangle, class X_t, class Y_t, class Z_t>
struct is_custom_sym_mat_vec_product_with_update_avail<
  Exec, A_t, Triangle, X_t, Y_t, Z_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(symmetric_matrix_vector_product(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<Triangle>(),
					       std::declval<X_t>(),
					       std::declval<Y_t>(),
					       std::declval<Z_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Overwriting hermitian matrix-vector product
template <class Exec, class A_t, class Triangle, class X_t, class Y_t, class = void>
struct is_custom_hermitian_mat_vec_product_avail : std::false_type {};

template <class Exec, class A_t, class Triangle, class X_t, class Y_t>
struct is_custom_hermitian_mat_vec_product_avail<
  Exec, A_t, Triangle, X_t, Y_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_vector_product(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<Triangle>(),
					       std::declval<X_t>(),
					       std::declval<Y_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Overwriting hermitian matrix-vector product with update
template <class Exec, class A_t, class Triangle, class X_t, class Y_t, class Z_t, class = void>
struct is_custom_hermitian_mat_vec_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Triangle, class X_t, class Y_t, class Z_t>
struct is_custom_hermitian_mat_vec_product_with_update_avail<
  Exec, A_t, Triangle, X_t, Y_t, Z_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(hermitian_matrix_vector_product(std::declval<Exec>(),
					       std::declval<A_t>(),
					       std::declval<Triangle>(),
					       std::declval<X_t>(),
					       std::declval<Y_t>(),
					       std::declval<Z_t>()))
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Overwriting triangular matrix-vector product: y := A * x
template <class Exec, class A_t, class Tri_t, class D_t, class X_t, class Y_t, class = void>
struct is_custom_tri_mat_vec_product_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class X_t, class Y_t>
struct is_custom_tri_mat_vec_product_avail<
  Exec, A_t, Tri_t, D_t, X_t, Y_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(triangular_matrix_vector_product
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Tri_t>(),
		std::declval<D_t>(),
		std::declval<X_t>(),
		std::declval<Y_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// Overwriting triangular matrix-vector product with update
template <class Exec, class A_t, class Tri_t, class D_t, class X_t, class Y_t, class Z_t, class = void>
struct is_custom_tri_mat_vec_product_with_update_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class X_t, class Y_t, class Z_t>
struct is_custom_tri_mat_vec_product_with_update_avail<
  Exec, A_t, Tri_t, D_t, X_t, Y_t, Z_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(triangular_matrix_vector_product
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Tri_t>(),
		std::declval<D_t>(),
		std::declval<X_t>(),
		std::declval<Y_t>(),
		std::declval<Z_t>()
		)
	       )
      >
    && ! impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

// In-place triangular matrix-vector product with update
template <class Exec, class A_t, class Tri_t, class D_t, class Y_t, class = void>
struct is_custom_tri_mat_vec_product_with_inplace_avail : std::false_type {};

template <class Exec, class A_t, class Tri_t, class D_t, class Y_t>
struct is_custom_tri_mat_vec_product_with_inplace_avail<
  Exec, A_t, Tri_t, D_t, Y_t,
  std::enable_if_t<
    std::is_void_v<
      decltype(triangular_matrix_vector_product
	       (std::declval<Exec>(),
		std::declval<A_t>(),
		std::declval<Tri_t>(),
		std::declval<D_t>(),
		std::declval<Y_t>()
		)
	       )
      >
    && !linalg::impl::is_inline_exec_v<Exec>
    >
  >
  : std::true_type{};

} // end anonymous namespace

namespace impl {

  template<class T>
  struct is_mdspan {
    static constexpr bool value = false;
  };

  template<class ElementType, class Extents, class Layout, class Accessor>
  struct is_mdspan<mdspan<ElementType, Extents, Layout, Accessor>> {
    // FIXME (mfh 2022/06/19) not quite enough -- the template
    // parameters also need to meet mdspan's requirements -- but this
    // is enough to resolve ambiguity between the overwriting +
    // ExecutionPolicy matrix_vector_product, and the non-overwriting
    // + no-ExecutionPolicy matrix_vector_product.
    static constexpr bool value = true;
  };

  template<class T>
  constexpr bool is_mdspan_v = is_mdspan<T>::value;

} // namespace impl

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A> >::is_always_unique())
)
void matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = std::common_type_t<
    std::common_type_t<
      std::common_type_t<SizeType_A, SizeType_x>,
      SizeType_y>>;
  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
    for (size_type j = 0; j < A.extent(1); ++j) {
      y(i) += A(i,j) * x(j);
    }
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
	 /* requires */ (! impl::is_mdspan_v<std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>> &&
			 Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A> >::is_always_unique())
)
void matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  constexpr bool use_custom = is_custom_mat_vec_product_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), decltype(x), decltype(y)>::value;

  if constexpr(use_custom) {
    matrix_vector_product(impl::map_execpolicy_with_check(exec), A, x, y);
  } else {
    matrix_vector_product(impl::inline_exec_t{}, A, x, y);
  }
}

template<class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y>
void matrix_vector_product(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  matrix_vector_product(impl::default_exec_t{}, A, x, y);
}

namespace impl {
  template<class Layout, class Extents>
  struct always_unique_mapping {
    static constexpr bool value = false;
  };

  template<class Layout, class SizeType, size_t ... Extents>
  struct always_unique_mapping<Layout, extents<SizeType, Extents...>> {
  private:
    using extents_type = extents<SizeType, Extents...>;
  public:
    static constexpr bool value = Layout::template mapping<extents_type>::is_always_unique();
  };

  template<class Layout, class Extents>
  constexpr bool always_unique_mapping_v = always_unique_mapping<Layout, Extents>::value;
} // namespace impl

// Updating general matrix-vector product: z := y + A * x

// FIXME (mfh 2022/06/19) Some work-around here for GCC 9 and/or macro insufficiencies.
MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         /* class SizeType_A,
	 size_t numRows_A,
	 size_t numCols_A,
	 */
	 class Extents_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         class SizeType_x, size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         class SizeType_z, size_t ext_z,
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A /* SizeType_A, numRows_A, numCols_A */> && Extents_A::rank() == 2)
)
void matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */, Layout_A, Accessor_A> A,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, extents<SizeType_z, ext_z>, Layout_z, Accessor_z> z)
{
  using size_type = std::common_type_t<
    std::common_type_t<
      std::common_type_t<typename Extents_A::size_type /* SizeType_A */, SizeType_x>,
      SizeType_y>,
    SizeType_z>;
  for (size_type i = 0; i < A.extent(0); ++i) {
    z(i) = y(i);
    for (size_type j = 0; j < A.extent(1); ++j) {
      z(i) += A(i,j) * x(j);
    }
  }
}

// FIXME (mfh 2022/06/19) Some work-around here for GCC 9 and/or macro insufficiencies.
MDSPAN_TEMPLATE_REQUIRES(
         class ExecutionPolicy,
         class ElementType_A,
	 class Extents_A,
         /* class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A, */
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
	 class Extents_z,
         /* class SizeType_z, ::std::size_t ext_z, */
         class Layout_z,
         class Accessor_z,
	 /* requires */ (! impl::is_mdspan_v<std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>> &&
			 Layout_A::template mapping<Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ >::is_always_unique() &&
			 Extents_A::rank() == 2 && Extents_z::rank() == 1)
)
void matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ , Layout_A, Accessor_A> A,
  mdspan<ElementType_x, extents<SizeType_x, ext_x> , Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, Extents_z /* extents<SizeType_z, ext_z> */ , Layout_z, Accessor_z> z)
{

  constexpr bool use_custom = is_custom_mat_vec_product_with_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), decltype(x), decltype(y), decltype(z)>::value;

  if constexpr(use_custom) {
    matrix_vector_product(impl::map_execpolicy_with_check(exec), A, x, y, z);
  } else {
    matrix_vector_product(impl::inline_exec_t{}, A, x, y, z);
  }
}

// FIXME (mfh 2022/06/19) Some work-around here for GCC 9 and/or macro insufficiencies.
MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	 class Extents_A,
	 /* class SizeType_A,
	 ::std::size_t numRows_A,
         ::std::size_t numCols_A, */
         class Layout_A,
         class Accessor_A,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         class SizeType_z, ::std::size_t ext_z,
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ > && Extents_A::rank() == 2)
)
void matrix_vector_product(
  mdspan<ElementType_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */, Layout_A, Accessor_A> A,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, extents<SizeType_z, ext_z>, Layout_z, Accessor_z> z)
{
  matrix_vector_product(impl::default_exec_t{}, A, x, y, z);
}


// Overwriting symmetric matrix-vector product: y := A * x

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A>>::is_always_unique())
)
void symmetric_matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = std::common_type_t<
      std::common_type_t<SizeType_A, SizeType_x>,
    SizeType_y>;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
  }

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      y(j) += A(j,j) * x(j);
      for (size_type i = j + size_type(1); i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += A_ij * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i < j; ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += A_ij * x(i);
      }
      y(j) += A(j,j) * x(j);
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y>
void symmetric_matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  constexpr bool use_custom = is_custom_sym_mat_vec_product_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), Triangle, decltype(x), decltype(y)>::value;

  if constexpr(use_custom) {
    symmetric_matrix_vector_product(impl::map_execpolicy_with_check(exec), A, t, x, y);
  } else {
    symmetric_matrix_vector_product(impl::inline_exec_t{}, A, t, x, y);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A>>::is_always_unique())
)
void symmetric_matrix_vector_product(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  symmetric_matrix_vector_product(impl::default_exec_t{}, A, t, x, y);
}


// Updating symmetric matrix-vector product: z := y + A * x

// FIXME (mfh 2022/06/19) Some work-around here for GCC 9 and/or macro insufficiencies.
MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	 class Extents_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         /* class SizeType_z, ::std::size_t ext_z, */
	 class Extents_z,
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A> && Extents_A::rank() == 2 && Extents_z::rank() == 1)
)
void symmetric_matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, Extents_A, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, Extents_z, Layout_z, Accessor_z> z)
{
  using size_type = std::common_type_t<
    std::common_type_t<
      std::common_type_t<typename Extents_A::size_type, SizeType_x>,
      SizeType_y>,
    typename Extents_z::size_type>;

  for (size_type i = 0; i < A.extent(0); ++i) {
    z(i) = y(i);
  }

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      z(j) += A(j,j) * x(j);
      for (size_type i = j + size_type(1); i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += A_ij * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i < j; ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += A_ij * x(i);
      }
      z(j) += A(j,j) * x(j);
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         class SizeType_z, ::std::size_t ext_z,
         class Layout_z,
         class Accessor_z>
auto symmetric_matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, extents<SizeType_z, ext_z>, Layout_z, Accessor_z> z) -> std::enable_if_t<!std::is_same_v<ExecutionPolicy, impl::inline_exec_t>>
{
  constexpr bool use_custom = is_custom_sym_mat_vec_product_with_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), Triangle, decltype(x), decltype(y), decltype(z)>::value;

  if constexpr(use_custom) {
    symmetric_matrix_vector_product(impl::map_execpolicy_with_check(exec), A, t, x, y, z);
  } else {
    symmetric_matrix_vector_product(impl::inline_exec_t{}, A, t, x, y, z);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	 class Extents_A,
	 /*
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
	 */
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
	 class Extents_z,
         /* class SizeType_z, ::std::size_t ext_z, */
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ > && Extents_A::rank() == 2 && Extents_z::rank() == 1)
)
void symmetric_matrix_vector_product(
  mdspan<ElementType_A,  Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ , Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, Extents_z /* extents<SizeType_z, ext_z> */ , Layout_z, Accessor_z> z)
{
  symmetric_matrix_vector_product(impl::default_exec_t{}, A, t, x, y, z);
}



// Overwriting Hermitian matrix-vector product: y := A * x

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A>>::is_always_unique())
)
void hermitian_matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = std::common_type_t<
      std::common_type_t<SizeType_A, SizeType_x>,
    SizeType_y>;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
  }

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      y(j) += impl::real_if_needed(A(j,j)) * x(j);
      for (size_type i = j + size_type(1); i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += impl::conj_if_needed(A_ij) * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i < j; ++i) {
        const auto A_ij = A(i,j);
        y(i) += A_ij * x(j);
        y(j) += impl::conj_if_needed(A_ij) * x(i);
      }
      y(j) += impl::real_if_needed(A(j,j)) * x(j);
    }
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A>>::is_always_unique())
)
void hermitian_matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  constexpr bool use_custom = is_custom_hermitian_mat_vec_product_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), Triangle, decltype(x), decltype(y)>::value;

  if constexpr(use_custom) {
    hermitian_matrix_vector_product(impl::map_execpolicy_with_check(exec), A, t, x, y);
  } else {
    hermitian_matrix_vector_product(impl::inline_exec_t{}, A, t, x, y);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A>>::is_always_unique())
)
void hermitian_matrix_vector_product(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  hermitian_matrix_vector_product(impl::default_exec_t{}, A, t, x, y);
}


// Updating Hermitian matrix-vector product: z := y + A * x

// FIXME (mfh 2022/06/19) Some work-around here for GCC 9 and/or macro insufficiencies.
MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	 class Extents_A,
	 /*
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
	 */
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
	 class Extents_z,
	 /* class SizeType_z, ::std::size_t ext_z, */
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ > && Extents_A::rank() == 2 && Extents_z::rank() == 1)
)
void hermitian_matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ , Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, Extents_z /* extents<SizeType_z, ext_z> */ , Layout_z, Accessor_z> z)
{
  using size_type = std::common_type_t<
    std::common_type_t<
      std::common_type_t<typename Extents_A::size_type /* SizeType_A */ , SizeType_x>,
      SizeType_y>,
    typename Extents_z::size_type /* SizeType_z */ >;

  for (size_type i = 0; i < A.extent(0); ++i) {
    z(i) = y(i);
  }

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      z(j) += impl::real_if_needed(A(j,j)) * x(j);
      for (size_type i = j + size_type(1); i < A.extent(0); ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += impl::conj_if_needed(A_ij) * x(i);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i < j; ++i) {
        const auto A_ij = A(i,j);
        z(i) += A_ij * x(j);
        z(j) += impl::conj_if_needed(A_ij) * x(i);
      }
      z(j) += impl::real_if_needed(A(j,j)) * x(j);
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         class SizeType_z, ::std::size_t ext_z,
         class Layout_z,
         class Accessor_z>
auto hermitian_matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, extents<SizeType_z, ext_z>, Layout_z, Accessor_z> z) -> std::enable_if_t<!std::is_same_v<ExecutionPolicy, impl::inline_exec_t>>
{
  constexpr bool use_custom = is_custom_hermitian_mat_vec_product_with_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)), decltype(A), Triangle, decltype(x), decltype(y), decltype(z)>::value;

  if constexpr(use_custom) {
    hermitian_matrix_vector_product(impl::map_execpolicy_with_check(exec), A, t, x, y, z);
  } else {
    hermitian_matrix_vector_product(impl::inline_exec_t{}, A, t, x, y, z);
  }
}

// FIXME (mfh 2022/06/19) Some work-around here for GCC 9 and/or macro insufficiencies.
MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	 class Extents_A,
	 /* class SizeType_A,
	 ::std::size_t numRows_A,
         ::std::size_t numCols_A, */
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
	 class Extents_z,
         /* class SizeType_z,
         ::std::size_t ext_z, */
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ > && Extents_A::rank() == 2)
)
void hermitian_matrix_vector_product(
  mdspan<ElementType_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ , Layout_A, Accessor_A> A,
  Triangle t,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, Extents_z /* extents<SizeType_z, ext_z> */ , Layout_z, Accessor_z> z)
{
  hermitian_matrix_vector_product(impl::default_exec_t{}, A, t, x, y, z);
}


// Overwriting triangular matrix-vector product: y := A * x

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A>>::is_always_unique())
)
void triangular_matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = std::common_type_t<
      std::common_type_t<SizeType_A, SizeType_x>,
    SizeType_y>;

  for (size_type i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
  }
  constexpr bool explicitDiagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const size_type i_lower = explicitDiagonal ? j : j + size_type(1);
      for (size_type i = i_lower; i < A.extent(0); ++i) {
        y(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        y(j) = y(j) + /* 1 times */ x(j);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const size_type i_upper = explicitDiagonal ? j + size_type(1) : j;
      for (size_type i = 0; i < i_upper; ++i) {
        y(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        y(j) = y(j) + /* 1 times */ x(j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y>
void triangular_matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{

  constexpr bool use_custom = is_custom_tri_mat_vec_product_avail<
    decltype(impl::map_execpolicy_with_check(exec)),
    decltype(A), decltype(t), decltype(d), decltype(x), decltype(y)
    >::value;

  if constexpr (use_custom) {
    triangular_matrix_vector_product(impl::map_execpolicy_with_check(exec), A, t, d, x, y);
  }
  else {
    triangular_matrix_vector_product(impl::inline_exec_t{},
				     A, t, d, x, y);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (Layout_A::template mapping<extents<SizeType_A, numRows_A, numCols_A>>::is_always_unique())
)
void triangular_matrix_vector_product(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  triangular_matrix_vector_product(impl::default_exec_t{}, A, t, d, x, y);
}


// Updating triangular matrix-vector product: z := y + A * x

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	 class Extents_A,
	 /* class SizeType_A,
	 ::std::size_t numRows_A,
         ::std::size_t numCols_A, */
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
	 class Extents_y,
         /* class SizeType_y,
         ::std::size_t ext_y, */
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
	 class Extents_z,
         /* class SizeType_z,
         ::std::size_t ext_z, */
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ > && Extents_A::rank() == 2)
)
void triangular_matrix_vector_product(
  impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ , Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y /* extents<SizeType_y, ext_y> */ , Layout_y, Accessor_y> y,
  mdspan<ElementType_z, Extents_z /* extents<SizeType_z, ext_z> */ , Layout_z, Accessor_z> z)
{
  using size_type = std::common_type_t<
    std::common_type_t<
      std::common_type_t<
	typename Extents_A::size_type /* SizeType_A */ ,
	SizeType_x>,
      typename Extents_y::size_type /* SizeType_y */ >,
    typename Extents_z::size_type /* SizeType_z */ >;

  for (size_type i = 0; i < A.extent(0); ++i) {
    z(i) = y(i);
  }
  constexpr bool explicitDiagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const size_type i_lower = explicitDiagonal ? j : j + size_type(1);
      for (size_type i = i_lower; i < A.extent(0); ++i) {
        z(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        z(j) = z(j) + /* 1 times */ x(j);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      const size_type i_upper = explicitDiagonal ? j + size_type(1) : j;
      for (size_type i = 0; i < i_upper; ++i) {
        z(i) += A(i,j) * x(j);
      }
      if constexpr (! explicitDiagonal) {
        z(j) = z(j) + /* 1 times */ x(j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
         class SizeType_z, ::std::size_t ext_z,
         class Layout_z,
         class Accessor_z>
auto triangular_matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
  mdspan<ElementType_z, extents<SizeType_z, ext_z>, Layout_z, Accessor_z> z) -> std::enable_if_t<!std::is_same_v<ExecutionPolicy, impl::inline_exec_t>>
{
  constexpr bool use_custom = is_custom_tri_mat_vec_product_with_update_avail<
    decltype(impl::map_execpolicy_with_check(exec)),
    decltype(A), decltype(t), decltype(d), decltype(x), decltype(y), decltype(z)
    >::value;

  if constexpr (use_custom) {
    triangular_matrix_vector_product(impl::map_execpolicy_with_check(exec), A, t, d, x, y, z);
  } else {
    triangular_matrix_vector_product(impl::inline_exec_t{}, A, t, d, x, y, z);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	 class Extents_A,
	 /* class SizeType_A,
	 ::std::size_t numRows_A,
         ::std::size_t numCols_A, */
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_x,
         class SizeType_x, ::std::size_t ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
	 class Extents_y,
         /* class SizeType_y,
         ::std::size_t ext_y, */
         class Layout_y,
         class Accessor_y,
         class ElementType_z,
	 class Extents_z,
         /* class SizeType_z,
         ::std::size_t ext_z, */
         class Layout_z,
         class Accessor_z,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ > && Extents_A::rank() == 2)
)
void triangular_matrix_vector_product(
  mdspan<ElementType_A, Extents_A /* extents<SizeType_A, numRows_A, numCols_A> */ , Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_x, extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
  mdspan<ElementType_y, Extents_y /* extents<SizeType_y, ext_y> */ , Layout_y, Accessor_y> y,
  mdspan<ElementType_z, Extents_z /* extents<SizeType_z, ext_z> */ , Layout_z, Accessor_z> z)
{
  triangular_matrix_vector_product(impl::default_exec_t{}, A, t, d, x, y, z);
}

// In-place triangular matrix-vector product: y := A * y

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
	     class SizeType_A,
	     ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_y,
         class SizeType_y,
         ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, extents<SizeType_A, numRows_A, numCols_A>>)
)
void triangular_matrix_vector_product(
  linalg::impl::inline_exec_t&& /* exec */,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A> , Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  using size_type = std::common_type_t<SizeType_A, SizeType_y>;
  constexpr bool explicitDiagonal =
    std::is_same_v<DiagonalStorage, explicit_diagonal_t>;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type k = 0; k < A.extent(1); ++k) {
      size_type j = A.extent(1) - (k + size_type(1));
      ElementType_y tmp = y(j);
      for (size_type i = j + size_type(1); i < A.extent(0); ++i) {
        y(i) += A(i,j) * tmp;
      }
      if constexpr (explicitDiagonal) {
        y(j) = y(j) * A(j,j);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      ElementType_y tmp = y(j);
      for (size_type i = 0; i < j; ++i) {
        y(i) += A(i,j) * tmp;
      }
      if constexpr (explicitDiagonal) {
        y(j) = y(j) * A(j,j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_A,
         class SizeType_A,
         ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_y,
         class SizeType_y, ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y>
void triangular_matrix_vector_product(
  ExecutionPolicy&& exec,
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  constexpr bool use_custom = is_custom_tri_mat_vec_product_with_inplace_avail<
    decltype(execpolicy_mapper(exec)),
    decltype(A), decltype(t), decltype(d), decltype(y)
    >::value;

  if constexpr(use_custom) {
    triangular_matrix_vector_product(execpolicy_mapper(exec), A, t, d, y);
  } else {
    triangular_matrix_vector_product(linalg::impl::inline_exec_t(), A, t, d, y);
  }
}

MDSPAN_TEMPLATE_REQUIRES(
         class ElementType_A,
    	 class SizeType_A,
    	 ::std::size_t numRows_A,
         ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle,
         class DiagonalStorage,
         class ElementType_y,
         class SizeType_y,
         ::std::size_t ext_y,
         class Layout_y,
         class Accessor_y,
         /* requires */ (impl::always_unique_mapping_v<Layout_A, extents<SizeType_A, numRows_A, numCols_A>>)
)
void triangular_matrix_vector_product(
  mdspan<ElementType_A, extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t,
  DiagonalStorage d,
  mdspan<ElementType_y, extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{
  triangular_matrix_vector_product(linalg::impl::default_exec_t(), A, t, d, y);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace MDSPAN_IMPL_PROPOSED_NAMESPACE
} // end namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_PRODUCT_HPP_
