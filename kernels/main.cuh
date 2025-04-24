#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>


const float ZNear = 0.3f;
const float ZFar = 1.1f;
const int ZPlanes = 256;

// This is the public interface of our cuda function, called directly in main.cpp
std::vector<cv::Mat> sweeping_plane_naive(cam const& ref, std::vector<cam> const& cam_vector, int window = 3);