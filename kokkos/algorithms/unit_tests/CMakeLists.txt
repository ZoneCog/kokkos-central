#Leave these here for now - I don't need transitive deps anyway
kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
kokkos_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
kokkos_include_directories(${KOKKOS_SOURCE_DIR}/core/unit_test/category_files)

set(ALGORITHM UnitTestMain.cpp)

foreach(Tag Threads;Serial;OpenMP;Cuda;HPX;HIP;SYCL;OpenMPTarget)
  string(TOUPPER ${Tag} DEVICE)
  string(TOLOWER ${Tag} dir)

  if(Kokkos_ENABLE_${DEVICE})
    set(dir ${CMAKE_CURRENT_BINARY_DIR}/${dir})
    file(MAKE_DIRECTORY ${dir})

    # ------------------------------------------
    # Sort
    # ------------------------------------------
    # Each of these inputs is an .hpp file.
    # Generate a .cpp file for each one that runs it on the current backend (Tag),
    # and add this .cpp file to the sources for UnitTest_RandomAndSort.
    set(ALGO_SORT_SOURCES)
    foreach(SOURCE_Input TestSort TestSortByKey TestSortCustomComp TestBinSortA TestBinSortB TestNestedSort)
      set(file ${dir}/${SOURCE_Input}.cpp)
      # Write to a temporary intermediate file and call configure_file to avoid
      # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
      file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n" "#include <${SOURCE_Input}.hpp>\n")
      configure_file(${dir}/dummy.cpp ${file})
      list(APPEND ALGO_SORT_SOURCES ${file})
    endforeach()

    # ------------------------------------------
    # Random
    # ------------------------------------------
    # do as above
    set(ALGO_RANDOM_SOURCES)
    foreach(SOURCE_Input TestRandom)
      set(file ${dir}/${SOURCE_Input}.cpp)
      file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n" "#include <${SOURCE_Input}.hpp>\n")
      configure_file(${dir}/dummy.cpp ${file})
      list(APPEND ALGO_RANDOM_SOURCES ${file})
    endforeach()
  endif()
endforeach()

# ------------------------------------------
# std set A
# ------------------------------------------
set(STDALGO_SOURCES_A)
foreach(Name StdReducers StdAlgorithmsConstraints RandomAccessIterator)
  list(APPEND STDALGO_SOURCES_A Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std set B
# ------------------------------------------
set(STDALGO_SOURCES_B)
foreach(Name StdAlgorithmsCommon StdAlgorithmsMinMaxElementOps)
  list(APPEND STDALGO_SOURCES_B Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std set C
# ------------------------------------------
set(STDALGO_SOURCES_C)
foreach(
  Name
  StdAlgorithmsCommon
  StdAlgorithmsLexicographicalCompare
  StdAlgorithmsForEach
  StdAlgorithmsFind
  StdAlgorithmsFindFirstOf
  StdAlgorithmsFindEnd
  StdAlgorithmsCount
  StdAlgorithmsEqual
  StdAlgorithmsAllAnyNoneOf
  StdAlgorithmsAdjacentFind
  StdAlgorithmsSearch
  StdAlgorithmsSearch_n
  StdAlgorithmsMismatch
  StdAlgorithmsMoveBackward
)
  list(APPEND STDALGO_SOURCES_C Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std set D
# ------------------------------------------
set(STDALGO_SOURCES_D)
foreach(
  Name
  StdAlgorithmsCommon
  StdAlgorithmsModOps
  StdAlgorithmsModSeqOps
  StdAlgorithmsReplace
  StdAlgorithmsReplaceIf
  StdAlgorithmsReplaceCopy
  StdAlgorithmsReplaceCopyIf
  StdAlgorithmsCopyIf
  StdAlgorithmsUnique
  StdAlgorithmsUniqueCopy
  StdAlgorithmsRemove
  StdAlgorithmsRemoveIf
  StdAlgorithmsRemoveCopy
  StdAlgorithmsRemoveCopyIf
  StdAlgorithmsRotate
  StdAlgorithmsRotateCopy
  StdAlgorithmsReverse
  StdAlgorithmsShiftLeft
  StdAlgorithmsShiftRight
)
  list(APPEND STDALGO_SOURCES_D Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std set E
# ------------------------------------------
set(STDALGO_SOURCES_E)
foreach(
  Name
  StdAlgorithmsCommon
  StdAlgorithmsIsSorted
  StdAlgorithmsIsSortedUntil
  StdAlgorithmsPartitioningOps
  StdAlgorithmsPartitionCopy
  StdAlgorithmsNumerics
  StdAlgorithmsAdjacentDifference
  StdAlgorithmsExclusiveScan
  StdAlgorithmsInclusiveScan
  StdAlgorithmsTransformUnaryOp
  StdAlgorithmsTransformExclusiveScan
  StdAlgorithmsTransformInclusiveScan
)
  list(APPEND STDALGO_SOURCES_E Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team Q
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_Q)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamInclusiveScan StdAlgorithmsTeamTransformInclusiveScan)
  list(APPEND STDALGO_TEAM_SOURCES_Q Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team P
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_P)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamExclusiveScan StdAlgorithmsTeamTransformExclusiveScan)
  list(APPEND STDALGO_TEAM_SOURCES_P Test${Name}.cpp)
endforeach()
if(KOKKOS_ENABLE_OPENMPTARGET) # FIXME_OPENMPTARGET
  list(REMOVE_ITEM STDALGO_TEAM_SOURCES_P TestStdAlgorithmsTeamExclusiveScan.cpp
       TestStdAlgorithmsTeamTransformExclusiveScan.cpp
  )
endif()

# ------------------------------------------
# std team M
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_M)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamTransformUnaryOp StdAlgorithmsTeamTransformBinaryOp
             StdAlgorithmsTeamGenerate StdAlgorithmsTeamGenerate_n StdAlgorithmsTeamSwapRanges
)
  list(APPEND STDALGO_TEAM_SOURCES_M Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team L
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_L)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamIsSorted StdAlgorithmsTeamIsSortedUntil
             StdAlgorithmsTeamIsPartitioned StdAlgorithmsTeamPartitionCopy StdAlgorithmsTeamPartitionPoint
)
  list(APPEND STDALGO_TEAM_SOURCES_L Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team I
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_I)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamUnique StdAlgorithmsTeamAdjacentDifference StdAlgorithmsTeamReduce
             StdAlgorithmsTeamTransformReduce
)
  list(APPEND STDALGO_TEAM_SOURCES_I Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team H
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_H)
foreach(
  Name
  StdAlgorithmsCommon
  StdAlgorithmsTeamCopy
  StdAlgorithmsTeamCopy_n
  StdAlgorithmsTeamCopyBackward
  StdAlgorithmsTeamCopyIf
  StdAlgorithmsTeamUniqueCopy
  StdAlgorithmsTeamRemove
  StdAlgorithmsTeamRemoveIf
  StdAlgorithmsTeamRemoveCopy
  StdAlgorithmsTeamRemoveCopyIf
)
  list(APPEND STDALGO_TEAM_SOURCES_H Test${Name}.cpp)
endforeach()

if(KOKKOS_ENABLE_OPENMPTARGET) # FIXME_OPENMPTARGET
  list(REMOVE_ITEM STDALGO_TEAM_SOURCES_H TestStdAlgorithmsTeamCopyIf.cpp TestStdAlgorithmsTeamUniqueCopy.cpp
       TestStdAlgorithmsTeamRemoveCopy.cpp TestStdAlgorithmsTeamRemoveCopyIf.cpp
  )
endif()

# ------------------------------------------
# std team G
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_G)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamMove StdAlgorithmsTeamMoveBackward StdAlgorithmsTeamShiftLeft
             StdAlgorithmsTeamShiftRight
)
  list(APPEND STDALGO_TEAM_SOURCES_G Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team F
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_F)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamReverse StdAlgorithmsTeamReverseCopy StdAlgorithmsTeamRotate
             StdAlgorithmsTeamRotateCopy
)
  list(APPEND STDALGO_TEAM_SOURCES_F Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team E
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_E)
foreach(
  Name
  StdAlgorithmsCommon
  StdAlgorithmsTeamFill
  StdAlgorithmsTeamFill_n
  StdAlgorithmsTeamReplace
  StdAlgorithmsTeamReplaceIf
  StdAlgorithmsTeamReplaceCopy
  StdAlgorithmsTeamReplaceCopyIf
)
  list(APPEND STDALGO_TEAM_SOURCES_E Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team D
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_D)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamMinElement StdAlgorithmsTeamMaxElement StdAlgorithmsTeamMinMaxElement)
  list(APPEND STDALGO_TEAM_SOURCES_D Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team C
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_C)
foreach(
  Name
  StdAlgorithmsCommon
  StdAlgorithmsTeamFind
  StdAlgorithmsTeamFindIf
  StdAlgorithmsTeamFindIfNot
  StdAlgorithmsTeamAllOf
  StdAlgorithmsTeamAnyOf
  StdAlgorithmsTeamNoneOf
  StdAlgorithmsTeamSearchN
)
  list(APPEND STDALGO_TEAM_SOURCES_C Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team B
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_B)
foreach(Name StdAlgorithmsCommon StdAlgorithmsTeamEqual StdAlgorithmsTeamSearch StdAlgorithmsTeamFindEnd
             StdAlgorithmsTeamFindFirstOf
)
  list(APPEND STDALGO_TEAM_SOURCES_B Test${Name}.cpp)
endforeach()

# ------------------------------------------
# std team A
# ------------------------------------------
set(STDALGO_TEAM_SOURCES_A)
foreach(
  Name
  StdAlgorithmsCommon
  StdAlgorithmsTeamAdjacentFind
  StdAlgorithmsTeamCount
  StdAlgorithmsTeamCountIf
  StdAlgorithmsTeamForEach
  StdAlgorithmsTeamForEachN
  StdAlgorithmsTeamLexicographicalCompare
  StdAlgorithmsTeamMismatch
)
  list(APPEND STDALGO_TEAM_SOURCES_A Test${Name}.cpp)
endforeach()

# FIXME_OPENMPTARGET - remove sort test as it leads to ICE with clang/16 and above at compile time.
if(KOKKOS_ENABLE_OPENMPTARGET AND KOKKOS_CXX_COMPILER_ID STREQUAL "Clang" AND KOKKOS_CXX_COMPILER_VERSION
                                                                              VERSION_GREATER_EQUAL 16.0.0
)
  list(REMOVE_ITEM ALGO_SORT_SOURCES TestSort.cpp)
endif()

# FIXME_OPENMPTARGET remove tests for OpenMPTarget because in these cases
# the impl needs to use either Kokkos or tailored reducers
# which results in runtime memory errors.
if(KOKKOS_ENABLE_OPENMPTARGET)
  list(REMOVE_ITEM STDALGO_TEAM_SOURCES_L TestStdAlgorithmsTeamIsPartitioned.cpp
       TestStdAlgorithmsTeamPartitionPoint.cpp TestStdAlgorithmsTeamPartitionCopy.cpp
  )
endif()

# FIXME_OPENMPTARGET need to remove tests for OpenMPTarget because
# in these cases the impl needs to use either Kokkos or
# tailored reducers which results in runtime memory errors.
if(KOKKOS_ENABLE_OPENMPTARGET)
  list(
    REMOVE_ITEM
    STDALGO_TEAM_SOURCES_C
    TestStdAlgorithmsTeamFind.cpp
    TestStdAlgorithmsTeamFindIf.cpp
    TestStdAlgorithmsTeamFindIfNot.cpp
    TestStdAlgorithmsTeamAllOf.cpp
    TestStdAlgorithmsTeamAnyOf.cpp
    TestStdAlgorithmsTeamNoneOf.cpp
    TestStdAlgorithmsTeamSearchN.cpp
  )
endif()

kokkos_add_executable_and_test(UnitTest_Sort SOURCES UnitTestMain.cpp TestStdAlgorithmsCommon.cpp ${ALGO_SORT_SOURCES})

kokkos_add_executable_and_test(UnitTest_Random SOURCES UnitTestMain.cpp ${ALGO_RANDOM_SOURCES})

# FIXME_OPENMPTARGET remove tests for OpenMPTarget
# causing failures for various reasons
if(KOKKOS_ENABLE_OPENMPTARGET)
  # the following use either Kokkos or tailored reducers
  # which results in runtime memory errors.
  list(REMOVE_ITEM STDALGO_TEAM_SOURCES_B TestStdAlgorithmsTeamFindEnd.cpp TestStdAlgorithmsTeamFindFirstOf.cpp
       TestStdAlgorithmsTeamSearch.cpp
  )

  list(REMOVE_ITEM STDALGO_TEAM_SOURCES_A TestStdAlgorithmsTeamAdjacentFind.cpp
       TestStdAlgorithmsTeamLexicographicalCompare.cpp TestStdAlgorithmsTeamMismatch.cpp
  )

  # this causes an illegal memory access if team_members_have_matching_result
  # is called
  list(REMOVE_ITEM STDALGO_TEAM_SOURCES_M TestStdAlgorithmsTeamTransformBinaryOp.cpp)
endif()

foreach(ID A;B;C;D;E)
  kokkos_add_executable_and_test(AlgorithmsUnitTest_StdSet_${ID} SOURCES UnitTestMain.cpp ${STDALGO_SOURCES_${ID}})
endforeach()

foreach(ID A;B;C;D;E;F;G;H;I;L;M;P;Q)
  kokkos_add_executable_and_test(
    AlgorithmsUnitTest_StdSet_Team_${ID} SOURCES UnitTestMain.cpp ${STDALGO_TEAM_SOURCES_${ID}}
  )
endforeach()

kokkos_add_executable(AlgorithmsUnitTest_StdAlgoCompileOnly SOURCES TestStdAlgorithmsCompileOnly.cpp)
