

#include <array_ref>
#include <iostream>

void test_layout_right()
{
  using dim_t = std::experimental::detail::extents_impl<0,size_t,10,20,0,40> ;
  using map_t = typename
    std::experimental::detail::layout_mapping
      < std::experimental::layout_right , dim_t >::type ;

  constexpr map_t m( dim_t(30) );

  static_assert( m.span() == 10*20*30*40 , "" );
  static_assert( m.stride(0) == 20*30*40 , "" );
  static_assert( m.stride(1) == 30*40 , "" );
  static_assert( m.stride(2) == 40 , "" );
  static_assert( m.stride(3) == 1 , "" );

  static_assert( m(0,0,0,0) == 0 , "" );
  static_assert( m(1,0,0,0) == 1*20*30*40 , "" );
  static_assert( m(0,1,0,0) == 1*30*40 , "" );
  static_assert( m(0,0,1,0) == 1*40 , "" );
  static_assert( m(0,0,0,1) == 1 , "" );

}


void test_layout_left()
{
  using dim_t = std::experimental::detail::extents_impl<0,size_t,10,20,0,40> ;
  using map_t = typename
    std::experimental::detail::layout_mapping
      < std::experimental::layout_left , dim_t >::type ;

  constexpr map_t m( dim_t(30) );

  static_assert( m.span() == 10*20*30*40 , "" );
  static_assert( m.stride(0) == 1 , "" );
  static_assert( m.stride(1) == 10 , "" );
  static_assert( m.stride(2) == 10*20 , "" );
  static_assert( m.stride(3) == 10*20*30 , "" );

  static_assert( m(0,0,0,0) == 0 , "" );
  static_assert( m(1,0,0,0) == 1 , "" );
  static_assert( m(0,1,0,0) == 1*10 , "" );
  static_assert( m(0,0,1,0) == 1*10*20 , "" );
  static_assert( m(0,0,0,1) == 1*10*20*30 , "" );

}


int main()
{
  test_layout_right();
  test_layout_left();
  return 0 ;
}

