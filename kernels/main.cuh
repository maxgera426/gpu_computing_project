#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include "../src/constants.hpp"



// This is the public interface of our cuda function, called directly in main.cpp
std::vector<cv::Mat> sweeping_plane_naive(cam const& ref, std::vector<cam> const& cam_vector, int window);
std::vector<cv::Mat> sweeping_plane_float_naive(cam const& ref, std::vector<cam> const& cam_vector, int window);
std::vector<cv::Mat> sweeping_plane_full_cam(cam const& ref, std::vector<cam> const& cam_vector, int window);
std::vector<cv::Mat> sweeping_plane_reduced_maxtrix(cam const& ref, std::vector<cam> const& cam_vector, int window);
std::vector<cv::Mat>  sweeping_plane_constant_mem(cam const& ref, std::vector<cam> const& cam_vector, int window);
