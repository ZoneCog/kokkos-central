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

#include <TestStdAlgorithmsCommon.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
#include <std_algorithms/impl/Kokkos_ValueWrapperForNoNeutralElement.hpp>
#include <std_algorithms/impl/Kokkos_IdentityReferenceUnaryFunctor.hpp>
#include <std_algorithms/impl/Kokkos_TransformInclusiveScan.hpp>
#endif
#include <utility>
#include <iomanip>

namespace Test {
namespace stdalgos {
namespace TransformIncScan {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct UnifDist;

template <>
struct UnifDist<double> {
  using dist_type = std::uniform_real_distribution<double>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(0.05, 1.2) { m_gen.seed(1034343); }

  double operator()() { return m_dist(m_gen); }
};

template <>
struct UnifDist<int> {
  using dist_type = std::uniform_int_distribution<int>;
  std::mt19937 m_gen;
  dist_type m_dist;

  UnifDist() : m_dist(1, 3) { m_gen.seed(1034343); }

  int operator()() { return m_dist(m_gen); }
};

template <class ViewType>
void fill_zero(ViewType view) {
  Kokkos::parallel_for(view.extent(0), FillZeroFunctor<ViewType>(view));
}

template <class ViewType>
void fill_view(ViewType dest_view, const std::string& name) {
  using value_type = typename ViewType::value_type;
  using exe_space  = typename ViewType::execution_space;

  const std::size_t ext = dest_view.extent(0);
  using aux_view_t      = Kokkos::View<value_type*, exe_space>;
  aux_view_t aux_view("aux_view", ext);
  auto v_h = create_mirror_view(Kokkos::HostSpace(), aux_view);

  UnifDist<value_type> randObj;

  if (name == "empty") {
    // no op
  }

  else if (name == "one-element") {
    KOKKOS_ASSERT(v_h.extent(0) == 1);
    v_h(0) = static_cast<value_type>(1);
  }

  else if (name == "two-elements-a") {
    KOKKOS_ASSERT(v_h.extent(0) == 2);
    v_h(0) = static_cast<value_type>(1);
    v_h(1) = static_cast<value_type>(2);
  }

  else if (name == "two-elements-b") {
    KOKKOS_ASSERT(v_h.extent(0) == 2);
    v_h(0) = static_cast<value_type>(2);
    v_h(1) = static_cast<value_type>(-1);
  }

  else if (name == "small-a") {
    KOKKOS_ASSERT(v_h.extent(0) == 9);
    v_h(0) = static_cast<value_type>(3);
    v_h(1) = static_cast<value_type>(1);
    v_h(2) = static_cast<value_type>(4);
    v_h(3) = static_cast<value_type>(1);
    v_h(4) = static_cast<value_type>(5);
    v_h(5) = static_cast<value_type>(9);
    v_h(6) = static_cast<value_type>(2);
    v_h(7) = static_cast<value_type>(6);
    v_h(8) = static_cast<value_type>(2);
  }

  else if (name == "small-b") {
    KOKKOS_ASSERT(v_h.extent(0) >= 6);
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
    v_h(5) = static_cast<value_type>(-2);
  }

  else if (name == "medium" || name == "large") {
    for (std::size_t i = 0; i < ext; ++i) {
      v_h(i) = randObj();
    }
  }

  else {
    FAIL() << "invalid choice";
  }

  Kokkos::deep_copy(aux_view, v_h);
  CopyFunctor<aux_view_t, ViewType> F1(aux_view, dest_view);
  Kokkos::parallel_for("copy", dest_view.extent(0), F1);
}

// I had to write my own because std::transform_inclusive_scan is ONLY found
// with std=c++17
template <class it1, class it2, class BopType, class UopType>
void my_host_transform_inclusive_scan(it1 first, it1 last, it2 dest,
                                      BopType bop, UopType uop) {
  if (first != last) {
    auto init = uop(*first);
    *dest     = init;
    while (++first < last) {
      init      = bop(uop(*first), init);
      *(++dest) = init;
    }
  }
}

template <class it1, class it2, class ValType, class BopType, class UopType>
void my_host_transform_inclusive_scan(it1 first, it1 last, it2 dest,
                                      BopType bop, UopType uop, ValType init) {
  if (first != last) {
    init  = bop(uop(*first), init);
    *dest = init;
    while (++first < last) {
      init      = bop(uop(*first), init);
      *(++dest) = init;
    }
  }
}

template <class ViewType1, class ViewType2, class... Args>
void verify_data(ViewType1 data_view,  // contains data
                 ViewType2 test_view,  // the view to test
                 Args... args /* by value on purpose*/) {
  //! always careful because views might not be deep copyable

  auto data_view_dc = create_deep_copyable_compatible_clone(data_view);
  auto data_view_h =
      create_mirror_view_and_copy(Kokkos::HostSpace(), data_view_dc);

  using gold_view_value_type = typename ViewType2::value_type;
  Kokkos::View<gold_view_value_type*, Kokkos::HostSpace> gold_h(
      "goldh", data_view.extent(0));
  my_host_transform_inclusive_scan(KE::cbegin(data_view_h),
                                   KE::cend(data_view_h), KE::begin(gold_h),
                                   args...);

  auto test_view_dc = create_deep_copyable_compatible_clone(test_view);
  auto test_view_h =
      create_mirror_view_and_copy(Kokkos::HostSpace(), test_view_dc);
  if (test_view_h.extent(0) > 0) {
    for (std::size_t i = 0; i < test_view_h.extent(0); ++i) {
      if (std::is_same_v<gold_view_value_type, int>) {
        ASSERT_EQ(gold_h(i), test_view_h(i));
      } else {
        const auto error = std::abs(gold_h(i) - test_view_h(i));
        ASSERT_LT(error, 1e-10) << i << " " << std::setprecision(15) << error
                                << static_cast<double>(test_view_h(i)) << " "
                                << static_cast<double>(gold_h(i));
      }
    }
  }
}

template <class ValueType>
struct TimesTwoUnaryFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a) const { return (a * ValueType(2)); }
};

template <class ValueType>
struct SumBinaryFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a + b);
  }
};

std::string value_type_to_string(int) { return "int"; }
std::string value_type_to_string(double) { return "double"; }

template <class Tag, class ValueType, class InfoType, class... Args>
void run_single_scenario(const InfoType& scenario_info,
                         Args... args /* by value on purpose*/) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t view_ext = std::get<1>(scenario_info);

  auto view_dest =
      create_view<ValueType>(Tag{}, view_ext, "transform_inclusive_scan");
  auto view_from =
      create_view<ValueType>(Tag{}, view_ext, "transform_inclusive_scan");
  fill_view(view_from, name);

  {
    fill_zero(view_dest);
    auto r = KE::transform_inclusive_scan(exespace(), KE::cbegin(view_from),
                                          KE::cend(view_from),
                                          KE::begin(view_dest), args...);
    ASSERT_EQ(r, KE::end(view_dest));
    verify_data(view_from, view_dest, args...);
  }

  {
    fill_zero(view_dest);
    auto r = KE::transform_inclusive_scan(
        "label", exespace(), KE::cbegin(view_from), KE::cend(view_from),
        KE::begin(view_dest), args...);
    ASSERT_EQ(r, KE::end(view_dest));
    verify_data(view_from, view_dest, args...);
  }

  {
    fill_zero(view_dest);
    auto r =
        KE::transform_inclusive_scan(exespace(), view_from, view_dest, args...);
    ASSERT_EQ(r, KE::end(view_dest));
    verify_data(view_from, view_dest, args...);
  }

  {
    fill_zero(view_dest);
    auto r = KE::transform_inclusive_scan("label", exespace(), view_from,
                                          view_dest, args...);
    ASSERT_EQ(r, KE::end(view_dest));
    verify_data(view_from, view_dest, args...);
  }

  Kokkos::fence();
}

template <class Tag, class ValueType, class InfoType, class... Args>
void run_single_scenario_inplace(const InfoType& scenario_info,
                                 Args... args /* by value on purpose*/) {
  const auto name            = std::get<0>(scenario_info);
  const std::size_t view_ext = std::get<1>(scenario_info);

  // since here we call the in-place operation, we need to use two views:
  // view1: filled according to scenario and is not modified
  // view2: filled according scenario and used for the in-place op
  // Therefore, after the op is done, view_2 should contain the
  // result of doing exclusive scan.
  // NOTE: view2 must be filled before every call to the algorithm
  // because the algorithm acts in place

  auto view_1 = create_view<ValueType>(Tag{}, view_ext,
                                       "transform_inclusive_scan_view_1");
  fill_view(view_1, name);

  auto view_2 = create_view<ValueType>(Tag{}, view_ext,
                                       "transform_inclusive_scan_view_2");

  {
    fill_view(view_2, name);
    auto r = KE::transform_inclusive_scan(exespace(), KE::cbegin(view_2),
                                          KE::cend(view_2), KE::begin(view_2),
                                          args...);
    ASSERT_EQ(r, KE::end(view_2));
    verify_data(view_1, view_2, args...);
  }

  {
    fill_view(view_2, name);
    auto r = KE::transform_inclusive_scan("label", exespace(),
                                          KE::cbegin(view_2), KE::cend(view_2),
                                          KE::begin(view_2), args...);
    ASSERT_EQ(r, KE::end(view_2));
    verify_data(view_1, view_2, args...);
  }

  {
    fill_view(view_2, name);
    auto r = KE::transform_inclusive_scan(exespace(), view_2, view_2, args...);
    ASSERT_EQ(r, KE::end(view_2));
    verify_data(view_1, view_2, args...);
  }

  {
    fill_view(view_2, name);
    auto r = KE::transform_inclusive_scan("label", exespace(), view_2, view_2,
                                          args...);
    ASSERT_EQ(r, KE::end(view_2));
    verify_data(view_1, view_2, args...);
  }

  Kokkos::fence();
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  const std::map<std::string, std::size_t> scenarios = {
      {"empty", 0},          {"one-element", 1}, {"two-elements-a", 2},
      {"two-elements-b", 2}, {"small-a", 9},     {"small-b", 13},
      {"medium", 1103},      {"large", 10513}};

  for (const auto& it : scenarios) {
    using uop_t = TimesTwoUnaryFunctor<ValueType>;
    using bop_t = SumBinaryFunctor<ValueType>;
    run_single_scenario<Tag, ValueType>(it, bop_t(), uop_t());
    run_single_scenario<Tag, ValueType>(it, bop_t(), uop_t(), ValueType{0});
    run_single_scenario<Tag, ValueType>(it, bop_t(), uop_t(), ValueType{1});
    run_single_scenario<Tag, ValueType>(it, bop_t(), uop_t(), ValueType{2});
    run_single_scenario<Tag, ValueType>(it, bop_t(), uop_t(), ValueType{-1});
    run_single_scenario<Tag, ValueType>(it, bop_t(), uop_t(), ValueType{-2});

    run_single_scenario_inplace<Tag, ValueType>(it, bop_t(), uop_t());
    run_single_scenario_inplace<Tag, ValueType>(it, bop_t(), uop_t(),
                                                ValueType{0});
    run_single_scenario_inplace<Tag, ValueType>(it, bop_t(), uop_t(),
                                                ValueType{2});
    run_single_scenario_inplace<Tag, ValueType>(it, bop_t(), uop_t(),
                                                ValueType{-2});
  }
}

#if !defined KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET
TEST(std_algorithms_numeric_ops_test, transform_inclusive_scan) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedThreeTag, double>();
  run_all_scenarios<DynamicTag, int>();
  run_all_scenarios<StridedThreeTag, int>();
}

template <class ValueType>
struct MultiplyFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a * b);
  }
};

TEST(std_algorithms_numeric_ops_test, transform_inclusive_scan_functor) {
  using value_type = KE::Impl::ValueWrapperForNoNeutralElement<int>;

  auto test_lambda = [&](auto& functor) {
    value_type value1;
    functor.init(value1);
    ASSERT_EQ(value1.val, 0);
    ASSERT_EQ(value1.is_initial, true);

    value_type value2;
    value2.val        = 1;
    value2.is_initial = false;
    functor.join(value1, value2);
    ASSERT_EQ(value1.val, 1);
    ASSERT_EQ(value1.is_initial, false);

    functor.init(value1);
    functor.join(value2, value1);
    ASSERT_EQ(value2.val, 1);
    ASSERT_EQ(value2.is_initial, false);

    functor.init(value2);
    functor.join(value2, value1);
    ASSERT_EQ(value2.val, 0);
    ASSERT_EQ(value2.is_initial, true);

    value1.val        = 3;
    value1.is_initial = false;
    value2.val        = 2;
    value2.is_initial = false;
    functor.join(value2, value1);
    ASSERT_EQ(value2.val, 6);
    ASSERT_EQ(value2.is_initial, false);
  };

  int dummy       = 0;
  using view_type = Kokkos::View<int*, exespace>;
  view_type dummy_view("dummy_view", 0);
  using unary_op_type = KE::Impl::StdNumericScanIdentityReferenceUnaryFunctor;
  {
    using functor_type =
        KE::Impl::ExeSpaceTransformInclusiveScanNoInitValueFunctor<
            exespace, int, int, view_type, view_type, MultiplyFunctor<int>,
            unary_op_type>;
    functor_type functor(dummy_view, dummy_view, {}, {});

    test_lambda(functor);
  }

  {
    using functor_type =
        KE::Impl::ExeSpaceTransformInclusiveScanWithInitValueFunctor<
            exespace, int, int, view_type, view_type, MultiplyFunctor<int>,
            unary_op_type>;
    functor_type functor(dummy_view, dummy_view, {}, {}, dummy);

    test_lambda(functor);
  }
}
#endif

}  // namespace TransformIncScan
}  // namespace stdalgos
}  // namespace Test
