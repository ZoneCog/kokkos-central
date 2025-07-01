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
// Contact: Jan Ciesko (jciesko@sandia.gov)
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>

#include <comm.hpp>

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using LocalView_t   = Kokkos::View<double***>;
using RemoteView_t  = Kokkos::View<double***, RemoteSpace_t>;
using HostView_t    = typename RemoteView_t::HostMirror;

struct CommHelper {
  MPI_Comm comm;
  // location in the grid
  int x, y, z;
  // Num MPI ranks in each dimension
  int nx, ny, nz;

  // Neightbor Ranks
  int up, down, left, right, front, back;

  // this process
  int me;
  int nranks;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    nx    = nranks;
    ny    = 1;
    nz    = 1;
    x     = me % nx;
    y     = (me / nx) % ny;
    z     = (me / nx / ny);
    left  = x == 0 ? -1 : me - 1;
    right = x == nx - 1 ? -1 : me + 1;
    down  = y == 0 ? -1 : me - nx;
    up    = y == ny - 1 ? -1 : me + nx;
    front = z == 0 ? -1 : me - nx * ny;
    back  = z == nz - 1 ? -1 : me + nx * ny;
#ifdef KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    printf("Me: %d MyNeighbors: %i %i %i %i %i %i\n", me, left, right, down, up,
           front, back);
#endif
  }
};

struct System {
  // Communicator
  CommHelper comm;

  // size of system
  int X, Y, Z;

  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;
  int lo_x, hi_x;

  // number of timesteps
  int N;

  // interval for print
  int I;

  // Temperature and delta Temperature
  RemoteView_t T;
  LocalView_t dT;
  HostView_t T_h;

  // Initial Temmperature
  double T0;

  // timestep width
  double dt;

  // thermal transfer coefficient
  double q;

  // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double sigma;

  // incoming power
  double P;

  // init_system
  System(MPI_Comm comm_) : comm(comm_) {
    // populate with defaults, set the rest in setup_subdomain.
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi               = X;
    Y_hi               = Y;
    Z_hi               = Z;
    lo_x               = 0;
    hi_x               = 0;
    N                  = 10000;
#if KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    I = 10;
#else
    I = N - 1;
#endif
    T0    = 0.0;
    dt    = 0.1;
    q     = 1.0;
    sigma = 1.0;
    P     = 1.0;
  }

  void setup_subdomain() {
    X_lo = Y_lo = Z_lo = 0;
    X_hi               = X;
    Y_hi               = Y;
    Z_hi               = Z;

    auto local_range = Kokkos::Experimental::get_local_range(X);
    lo_x             = local_range.first;
    hi_x             = local_range.second;

#if KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    printf("My Domain: %i (%i %i %i) (%i %i %i)\n", comm.me, lo_x, Y_lo, Z_lo,
           hi_x, Y_hi, Z_hi);
#endif
    T   = RemoteView_t("System::T", X, Y, Z);
    T_h = HostView_t("Host::T", T.extent(0), Y, Z);
    dT  = LocalView_t("System::dT", T.extent(0), Y, Z);

    Kokkos::deep_copy(T_h, T0);
    Kokkos::deep_copy(T, T_h);
  }

  void print_help() {
    printf("Options (default):\n");
    printf("  -X IARG: (%i) num elements in the X direction\n", X);
    printf("  -Y IARG: (%i) num elements in the Y direction\n", Y);
    printf("  -Z IARG: (%i) num elements in the Z direction\n", Z);
    printf("  -N IARG: (%i) num timesteps\n", N);
    printf("  -I IARG: (%i) print interval\n", I);
    printf("  -T0 FARG: (%lf) initial temperature\n", T0);
    printf("  -dt FARG: (%lf) timestep size\n", dt);
    printf("  -q FARG: (%lf) thermal conductivity\n", q);
    printf("  -sigma FARG: (%lf) thermal radiation\n", sigma);
    printf("  -P FARG: (%lf) incoming power\n", P);
  }

  // check command line args
  bool check_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0) {
        print_help();
        return false;
      }
    }
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-X") == 0) X = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Y") == 0) Y = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-Z") == 0) Z = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-I") == 0) I = atoi(argv[i + 1]);
      if (strcmp(argv[i], "-T0") == 0) T0 = atof(argv[i + 1]);
      if (strcmp(argv[i], "-dt") == 0) dt = atof(argv[i + 1]);
      if (strcmp(argv[i], "-q") == 0) q = atof(argv[i + 1]);
      if (strcmp(argv[i], "-sigma") == 0) sigma = atof(argv[i + 1]);
      if (strcmp(argv[i], "-P") == 0) P = atof(argv[i + 1]);
    }
    setup_subdomain();
    return true;
  }

  // compute both inner and outer updates. This function is suitable for both.
  struct ComputeDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    // Heat conduction on body (local accesses)
    if (x > 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < X - 1) dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < Y - 1) dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < Z - 1) dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction on dim0 boarder (remote accesses)
    if (x == 0 && lo_x != 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x == (hi_x - 1) && hi_x != X) dT_xyz += q * (T(x + 1, y, z) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // Thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (X - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0) ? 1 : 0) + (y == (Y - 1) ? 1 : 0) +
                       ((z == 0) ? 1 : 0) + (z == (Z - 1) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }

  void compute_dT() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeDT, int>;
    Kokkos::parallel_for(
        "ComputeDT",
        Kokkos::Experimental::require(
            policy_t({lo_x, 0, 0}, {hi_x, Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        *this);
    Kokkos::fence();
  }

  struct updateT {
    RemoteView_t T;
    LocalView_t dT;
    double dt;
    updateT(RemoteView_t T_, LocalView_t dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double& sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT(x, y, z);
    }
  };

  double update_T() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    double my_T;
    Kokkos::parallel_reduce(
        "ComputeT",
        Kokkos::Experimental::require(
            policy_t({lo_x, 0, 0}, {hi_x, Y, Z}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        updateT(T, dT, dt), my_T);
    double sum_T;
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
    return sum_T;
  }

  // run_time_loops
  void timestep() {
    Kokkos::Timer timer;
    double time_a, time_b, time_c, time_update, time_compute, time_all;
    time_update = time_compute = 0.0;

    double time_z = timer.seconds();

    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0; /* stop heat in halfway through */
      time_a = timer.seconds();
      compute_dT();
      RemoteSpace_t().fence();
      time_b       = timer.seconds();
      double T_ave = update_T();
      time_c       = timer.seconds();
      time_compute += time_b - time_a;
      time_update += time_c - time_b;
      T_ave /= X * Y * Z;
#ifdef KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
      if ((t % I == 0 || t == N) && (comm.me == 0))
        printf("Timestep: %i/%i t_avg: %lf\n", t, N, T_ave);
#endif
    }

    time_all = timer.seconds() - time_z;

    if (comm.me == 0) {
      printf("heat3D,KokkosRemoteSpaces,%i,%lf,%lf,%lf,%i,%lf,%lf,%lf\n",
             comm.nranks, time_compute / N, double(0), time_update / N, X,
             double(2 * T.span() * sizeof(double)) / 1024 / 1024, time_all / N,
             time_all);
    }
  }
};

int main(int argc, char* argv[]) {
  comm_init(argc, argv);
  Kokkos::initialize(argc, argv);
  {
    System sys(MPI_COMM_WORLD);
    if (sys.check_args(argc, argv)) sys.timestep();
  }
  Kokkos::finalize();
  comm_fini();
  return 0;
}
