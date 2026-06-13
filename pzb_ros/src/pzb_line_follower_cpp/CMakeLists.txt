cmake_minimum_required(VERSION 3.8)
project(pzb_line_follower_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# IMPORTANT: do NOT enable -ffast-math — bit-level FP fidelity vs the Python/numpy
# reference matters for the rosbag parity comparison (the adaptive-threshold feedback
# loop amplifies any divergence). -O2 is fine.
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O2 -Wall -Wextra -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)

include_directories(include)

# ── Detector library (reused by the node and the offline comparator) ──────────
add_library(center_line_detector src/center_line_detector.cpp)
target_include_directories(center_line_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(center_line_detector ${OpenCV_LIBS})

# ── bag reader helper (CDR Image parse + db3 access; ROS-free) ────────────────
add_library(bag_reader src/bag_reader.cpp)
target_include_directories(bag_reader PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bag_reader ${OpenCV_LIBS} SQLite3::SQLite3)

# ── Phase A: detector-only dump (offline, for parity vs the Python detector) ──
add_executable(detector_dump src/detector_dump_main.cpp)
target_link_libraries(detector_dump center_line_detector bag_reader ${OpenCV_LIBS})

# ── Follower node core + ROS node + comparator — Phase B (enabled below) ──────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/follower_core.cpp)
  add_library(follower_core src/follower_core.cpp)
  target_include_directories(follower_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(follower_core center_line_detector ${OpenCV_LIBS})

  add_executable(line_follower_node src/line_follower_node.cpp)
  target_link_libraries(line_follower_node follower_core center_line_detector ${OpenCV_LIBS})
  ament_target_dependencies(line_follower_node
    rclcpp sensor_msgs std_msgs geometry_msgs cv_bridge OpenCV)

  add_executable(replay_follower src/replay_follower_main.cpp)
  target_link_libraries(replay_follower follower_core center_line_detector bag_reader
    ${OpenCV_LIBS} SQLite3::SQLite3)

  install(TARGETS line_follower_node replay_follower DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS detector_dump DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
