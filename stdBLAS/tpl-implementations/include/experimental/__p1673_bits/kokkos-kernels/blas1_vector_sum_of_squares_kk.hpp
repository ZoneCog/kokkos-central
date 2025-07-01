
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_SUM_OF_SQUARES_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_VEC_SUM_OF_SQUARES_HPP_

#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

template<class ExecSpace,
         class ElementType,
         std::experimental::extents<>::size_type ext0,
         class Layout,
         class Scalar>
std::experimental::linalg::sum_of_squares_result<Scalar>
vector_sum_of_squares(kokkos_exec<ExecSpace> /*kexe*/,
		      std::experimental::mdspan<
		      ElementType,
		      std::experimental::extents<ext0>,
		      Layout,
		      std::experimental::default_accessor<ElementType>> x,
		      std::experimental::linalg::sum_of_squares_result<Scalar> init)
{

  Impl::signal_kokkos_impl_called("vector_sum_of_squares");

  auto x_view = Impl::mdspan_to_view(x);
  std::experimental::linalg::sum_of_squares_result<Scalar> result;

  using arithm_traits = Kokkos::Details::ArithTraits<ElementType>;

  Scalar scaling_factor = {};
  Kokkos::Max<Scalar> max_reducer(scaling_factor);
  Kokkos::parallel_reduce( Kokkos::RangePolicy(ExecSpace(), 0, x_view.extent(0)),
			   KOKKOS_LAMBDA (const std::size_t i, Scalar & lmax){
			     const auto val = arithm_traits::abs(x_view(i));
			     max_reducer.join(lmax, val);
			   },
			   max_reducer);
  // no fence needed since reducing into scalar
  result.scaling_factor = std::max(scaling_factor, init.scaling_factor);

  Scalar ssq = {};
  Kokkos::parallel_reduce(Kokkos::RangePolicy(ExecSpace(), 0, x_view.extent(0)),
			  KOKKOS_LAMBDA (const std::size_t i, Scalar & update){
			    const auto tmp = arithm_traits::abs(x_view(i))/result.scaling_factor;
			    update += tmp*tmp;
			  }, ssq);
  // no fence needed since reducing into scalar

  result.scaled_sum_of_squares = ssq
    + (init.scaling_factor*init.scaling_factor*init.scaled_sum_of_squares)/(scaling_factor*scaling_factor);

  return result;
}

}
#endif
