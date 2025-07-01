#define KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Core.hpp>
#include <TestAttentionAllocation.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    
    int result = Test::run_attention_allocation_tests();
    
    Kokkos::finalize();
    return result;
}