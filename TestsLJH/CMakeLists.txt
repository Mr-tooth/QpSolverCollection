cmake_minimum_required(VERSION 3.20)
project(TestQP)

#----Test for Boost package usage
set(Boost_LIB_PREFIX "lib")
set(BOOST_LIBRARYDIR "D:/Program Files True/BOOST/boost_1_73_0/stage/lib")
find_package(Boost COMPONENTS unit_test_framework timer)

#----Basic test
find_package(qp_solver_collection REQUIRED)
message(STATUS "BUILD_SHARED_LIBS is ${BUILD_SHARED_LIBS}")
message(STATUS "qp_solver_collection_LIBRARIES is ${qp_solver_collection_LIBRARIES}")
add_executable(testQP TestSampleQPLJH.cpp)
target_link_libraries(testQP PUBLIC ${qp_solver_collection_LIBRARIES})
target_compile_options(testQP PUBLIC /MT)