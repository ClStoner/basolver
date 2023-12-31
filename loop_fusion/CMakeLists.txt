cmake_minimum_required(VERSION 2.8.3)
project(loop_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
    )

find_package(OpenCV)
message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
find_package(Ceres REQUIRED)
message(WARNING "Ceres_VERSION: ${Ceres_VERSION}")
find_package(Eigen3)
message(WARNING "Eigen3_VERSION: ${Eigen3_VERSION}")
find_package(yaml-cpp REQUIRED)

include_directories(
    ${catkin_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}  
    ${EIGEN3_INCLUDE_DIR}
    ${YAML_INCLUDE_DIRS})

catkin_package()

add_executable(loop_node
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    )

target_link_libraries(loop_node 
    ${catkin_LIBRARIES}  
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES}
    ${YAML_CPP_LIBRARIES}) 
