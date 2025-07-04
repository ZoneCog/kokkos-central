pipeline {
    agent none

    environment {
        CCACHE_DIR = '/tmp/ccache'
        CCACHE_MAXSIZE = '5G'
        CCACHE_CPP2 = 'true'
    }

    options {
        disableConcurrentBuilds(abortPrevious: true)
        timeout(time: 6, unit: 'HOURS')
    }

    triggers {
        issueCommentTrigger('.*test this please.*')
    }

    stages {
        stage('Clang-Format') {
            agent {
                dockerfile {
                    filename 'Dockerfile.clang'
                    dir 'scripts/docker'
                    label 'nvidia-docker || docker'
                    args '-v /tmp/ccache.kokkos:/tmp/ccache'
                }
            }
            steps {
                sh './scripts/docker/check_format_cpp.sh'
            }
        }
        stage('Build-1') {
            parallel {
                stage('C++20-Modules-Clang-19') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.modules'
                            dir 'scripts/docker'
                            label 'docker'
                        }
                    }
                    steps {
                        sh '''rm -rf build && \
                              cmake \
                                -B build \
                                -GNinja \
                                -DCMAKE_CXX_COMPILER=clang++-19 \
                                -DCMAKE_CXX_FLAGS="-Werror" \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_EXPERIMENTAL_CXX20_MODULES=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_EXAMPLES=ON \
                                -DKokkos_ENABLE_SERIAL=ON && \
                              cmake --build build --target install -j 8 && \
                              ctest --test-dir build --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('GCC-8.4.0') {
                    agent {
                         dockerfile {
                             filename 'Dockerfile.gcc'
                             dir 'scripts/docker'
                             label 'docker'
                         }
                     }
                    environment {
                        OMP_NUM_THREADS = 8
                        OMP_NESTED = 'true'
                        OMP_MAX_ACTIVE_LEVELS = 3
                        OMP_PROC_BIND = 'true'
                    }
                    steps {
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_OPENMP=ON \
                                -DKokkos_ENABLE_LIBDL=OFF \
                                -DKokkos_ENABLE_LIBQUADMATH=ON \
                                -DKokkos_ENABLE_SERIAL=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose && gcc -I$PWD/../core/src/ ../core/unit_test/tools/TestCInterface.c'''
                    }
                    post {
                        always {
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('HIP-ROCm-5.7-CXX20') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.hipcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-22.04:5.7.1-complete@sha256:fc6abb843a4cb2b3e5d8e9225ed0db1450e063dbcc347f44b43252264134485d'
                            label 'rocm-docker'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DBUILD_SHARED_LIBS=ON \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER=hipcc \
                                -DCMAKE_CXX_FLAGS="-Werror -Wno-unused-command-line-argument" \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_HIP=ON \
                              .. && \
                              make -j16 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('CUDA-11.0-NVCC-RDC') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=nvcr.io/nvidia/cuda:11.0.3-devel-ubuntu20.04@sha256:10ab0f09fcdc796b4a2325ef1bce8f766f4a3500eab5a83780f80475ae26c7a6 --build-arg ADDITIONAL_PACKAGES="g++-8 gfortran clang" --build-arg CMAKE_VERSION=3.17.3'
                            label 'nvidia-docker && (volta || ampere)'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        OMP_NUM_THREADS = 8
                        // Nested OpenMP does not work for this configuration,
                        // so disabling it
                        OMP_MAX_ACTIVE_LEVELS = 1
                        OMP_PLACES = 'threads'
                        OMP_PROC_BIND = 'spread'
                        NVCC_WRAPPER_DEFAULT_COMPILER = 'g++-8'
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf install && mkdir -p install && \
                              rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DCMAKE_CXX_COMPILER=g++-8 \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_OPENMP=OFF \
                                -DKokkos_ENABLE_CUDA=ON \
                                -DKokkos_ENABLE_CUDA_LAMBDA=OFF \
                                -DKokkos_ENABLE_CUDA_UVM=ON \
                                -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                \
                                -DCMAKE_INSTALL_PREFIX=${PWD}/../install \
                              .. && \
                              make -j8 install && \
                              cd .. && \
                              rm -rf build-tests && mkdir -p build-tests && cd build-tests && \
                              export CMAKE_PREFIX_PATH=${PWD}/../install && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                -DCMAKE_CXX_COMPILER=$WORKSPACE/bin/nvcc_wrapper \
                                -DCMAKE_CXX_FLAGS="-Werror --Werror=all-warnings -Xcudafe --diag_suppress=940" \
                                -DCMAKE_EXE_LINKER_FLAGS="-Xnvlink -suppress-stack-size-warning" \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DKokkos_INSTALL_TESTING=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose && \
                              cd ../example/build_cmake_installed && \
                              rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_CXX_COMPILER=g++-8 \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DCMAKE_CXX_STANDARD=17 \
                              .. && \
                              make -j8 && ctest --verbose && \
                              cd ../.. && \
                              cmake -B build_cmake_installed_different_compiler/build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-Werror -DCMAKE_CXX_STANDARD=17 build_cmake_installed_different_compiler && \
                              cmake --build build_cmake_installed_different_compiler/build --target all && \
                              cmake --build build_cmake_installed_different_compiler/build --target test'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build-tests/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
            }
        }
        stage('Build-2') {
            parallel {
                stage('OPENACC-NVHPC-CUDA-12.2') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvhpc'
                            dir 'scripts/docker'
                            label 'nvidia-docker && volta && large_images'
                            args '--env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        NVHPC_CUDA_HOME = '/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2'
                    }
                    steps {
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              /opt/cmake/bin/cmake \
                                -DCMAKE_CXX_COMPILER=nvc++ \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DCMAKE_CXX_FLAGS=-Werror \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_OPENACC=ON \
                                -DKokkos_ARCH_VOLTA70=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }

                }
                stage('CUDA-12.2-NVHPC-AS-HOST-COMPILER') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvhpc'
                            dir 'scripts/docker'
                            label 'nvidia-docker && large_images && volta'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        OMP_NUM_THREADS = 8
                        // Nested OpenMP does not work for this configuration,
                        // so disabling it
                        OMP_MAX_ACTIVE_LEVELS = 1
                        OMP_PLACES = 'threads'
                        OMP_PROC_BIND = 'spread'
                        NVHPC_CUDA_HOME = '/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2'
                    }
                    steps {
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              /opt/cmake/bin/cmake \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER=nvc++ \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DCMAKE_CXX_FLAGS="-Werror --diag_suppress=implicit_return_from_non_void_function" \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_CUDA=ON \
                                -DKokkos_ENABLE_OPENMP=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }

                }
                stage('SYCL-OneAPI') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.sycl'
                            dir 'scripts/docker'
                            label 'nvidia-docker && ampere'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                -DCMAKE_CXX_COMPILER=icpx \
                                -DCMAKE_CXX_FLAGS="-fsycl-device-code-split=per_kernel -fp-model=precise -Wno-deprecated-declarations -Werror -Wno-gnu-zero-variadic-macro-arguments -Wno-unknown-cuda-version -Wno-sycl-target" \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ARCH_AMPERE80=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
                                -DKokkos_ENABLE_EXAMPLES=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DoneDPL_ROOT=/opt/intel/oneapi/dpl/2022.7 \
                                -DKokkos_ENABLE_SYCL=ON \
                                -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON \
                                -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON \
                                -DCMAKE_CXX_STANDARD=17 \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('HIP-ROCm-5.3') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.hipcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-20.04:5.3.3-complete@sha256:bac114b9d09e61d88b45fbeb40a15a315c2a78a203223c9b4ed7263b05ff3977'
                            label 'rocm-docker '
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        OMP_NUM_THREADS = 8
                        OMP_MAX_ACTIVE_LEVELS = 3
                        OMP_PLACES = 'threads'
                        OMP_PROC_BIND = 'spread'
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/llvm.conf && ldconfig'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Debug \
                                -DCMAKE_CXX_COMPILER=hipcc \
                                -DCMAKE_CXX_FLAGS="-Werror -Wno-unused-command-line-argument -DNDEBUG" \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_HIP=ON \
                                -DKokkos_ENABLE_OPENMP=ON \
                                -DKokkos_ENABLE_IMPL_MDSPAN=OFF \
                                -DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON \
                              .. && \
                              make -j16 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('HIP-ROCm-6.2-amdclang-CXX20') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.hipcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-24.04:6.2-complete@sha256:c7049ac3ae8516c7b230deec6dc6dd678a0b3f7215d5a7f7fe2f2b71880b62f8 --build-arg ADDITIONAL_PACKAGES="clang-tidy"'
                            label 'rocm-docker'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        // FIXME Test returns a wrong value
                        GTEST_FILTER = '-hip_hostpinned.view_allocation_large_rank'
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DBUILD_SHARED_LIBS=ON \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
                                -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-warnings-as-errors=*" \
                                -DCMAKE_PREFIX_PATH=/opt/rocm/lib \
                                -DCMAKE_CXX_FLAGS="-Werror -Wno-unused-command-line-argument" \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_HIP=ON \
                              .. && \
                              make -j16 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
/*
                stage('OPENMPTARGET-ROCm-5.2') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.hipcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-20.04:5.2'
                            label 'rocm-docker && vega && AMD_Radeon_Instinct_MI60'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        OMP_NUM_THREADS = 8
                        OMP_MAX_ACTIVE_LEVELS = 3
                        OMP_PLACES = 'threads'
                        OMP_PROC_BIND = 'spread'
                        LC_ALL = 'C'
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/llvm.conf && ldconfig'
                        sh '''rm -rf build && \
                              cmake \
                                -Bbuild \
                                -DCMAKE_BUILD_TYPE=Debug \
                                -DCMAKE_CXX_COMPILER=amdclang++ \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_OPENMPTARGET=ON \
                                -DKokkos_ENABLE_OPENMP=ON \
                                -DKokkos_ARCH_AMD_GFX906=ON \
                              && \
                              cmake --build build --parallel ${BUILD_JOBS} && \
                              cd build && ctest --no-compress-output -T Test --output-on-failure
                        '''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                        }
                    }
                }
*/
                stage('OPENMPTARGET-Clang') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.openmptarget'
                            dir 'scripts/docker'
                            label 'nvidia-docker && volta'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER=clang++ \
                                -DCMAKE_CXX_FLAGS="-Wno-unknown-cuda-version -Werror -Wno-undefined-internal -Wno-pass-failed" \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_TUNING=ON \
                                -DKokkos_ENABLE_OPENMPTARGET=ON \
                                -DKokkos_ARCH_VOLTA70=ON \
                                -DCMAKE_CXX_STANDARD=17 \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('CUDA-11.8-Clang-15') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvcc'
                            dir 'scripts/docker'
                            label 'nvidia-docker && volta'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                            additionalBuildArgs '--build-arg BASE=nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04 --build-arg ADDITIONAL_PACKAGES="clang-15 clang-tidy-15"'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                -DCMAKE_CXX_COMPILER=clang++-15 \
                                -DCMAKE_CXX_CLANG_TIDY="clang-tidy-15;-warnings-as-errors=*" \
                                -DCMAKE_CXX_FLAGS="-Werror -Wno-unknown-cuda-version -Wno-pass-failed" \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_CUDA=ON \
                                -DKokkos_ENABLE_TUNING=ON \
                                -DKokkos_ARCH_VOLTA70=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('CUDA-12.5.1-Clang-17-RDC') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvcc'
                            dir 'scripts/docker'
                            label 'nvidia-docker && volta'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                            additionalBuildArgs '--build-arg BASE=nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu24.04 --build-arg ADDITIONAL_PACKAGES="clang-17 clang-tidy-17"'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                -DCMAKE_CXX_COMPILER=clang++-17 \
                                -DCMAKE_CXX_CLANG_TIDY="clang-tidy-17;-warnings-as-errors=*" \
                                -DCMAKE_CXX_FLAGS="-Werror -Wno-unknown-cuda-version -Wno-pass-failed" \
                                -DCMAKE_CXX_STANDARD=20 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_CUDA=ON \
                                -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
                                -DKokkos_ENABLE_TUNING=ON \
                                -DKokkos_ARCH_VOLTA70=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('CUDA-11.6-NVCC-DEBUG') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04@sha256:d95d54bc231f8aea7fda79f60da620324584b20ed31a8ebdb0686cffd34dd405'
                            label 'nvidia-docker && (volta || ampere)'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DBUILD_SHARED_LIBS=ON \
                                -DCMAKE_BUILD_TYPE=Debug \
                                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                -DCMAKE_CXX_COMPILER=$WORKSPACE/bin/nvcc_wrapper \
                                -DCMAKE_CXX_FLAGS="-Werror -Werror=all-warnings" \
                                -DCMAKE_CXX_STANDARD=17 \
                                -DKokkos_ARCH_NATIVE=ON \
                                -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
                                -DKokkos_ENABLE_DEBUG=ON \
                                -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
                                -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DKokkos_ENABLE_BENCHMARKS=ON \
                                -DKokkos_ENABLE_CUDA=ON \
                                -DKokkos_ENABLE_LIBDL=OFF \
                                -DKokkos_ENABLE_OPENMP=ON \
                                -DKokkos_ENABLE_IMPL_MDSPAN=OFF \
                                -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON \
                              .. && \
                              make -j8 && ctest --no-compress-output -T Test --verbose && \
                              cd ../example/build_cmake_in_tree && \
                              rm -rf build && mkdir -p build && cd build && \
                              cmake -DCMAKE_CXX_STANDARD=17 .. && make -j8 && ctest --verbose'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
                        }
                    }
                }
                stage('CUDA-11.7-NVCC') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=nvcr.io/nvidia/cuda:11.7.1-devel-ubuntu20.04@sha256:fc997521e612899a01dce92820f5f5a201dd943ebfdc3e49ba0706d491a39d2d'
                            label 'nvidia-docker && volta'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              ../gnu_generate_makefile.bash \
                                --with-options=compiler_warnings \
                                --cxxflags="-Werror -Werror=all-warnings" \
                                --cxxstandard=c++17 \
                                --with-cuda \
                                --with-cuda-options=enable_lambda \
                                --arch=Volta70 \
                              && \
                              make test -j8'''
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                        }
                    }
                }
            }
        }
    }
}
