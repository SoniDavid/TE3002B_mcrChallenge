cmake_minimum_required(VERSION 3.8)
project(pzb_traffic_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O2 -Wall -Wextra -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(color_detector_node src/color_detector_node.cpp)
target_include_directories(color_detector_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(color_detector_node ${OpenCV_LIBS})
ament_target_dependencies(color_detector_node rclcpp sensor_msgs std_msgs cv_bridge OpenCV)

install(TARGETS color_detector_node DESTINATION lib/${PROJECT_NAME})

ament_package()
