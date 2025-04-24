#include "main.cuh"
#include <cstdio>
#include <cmath>

// Those functions are an example on how to call cuda functions from the main.cpp
__global__ void sweeping_plane_naive(
	//Reference data
	double* ref_K_inv, double* ref_R_inv, double* ref_t_inv,
	int ref_width, int ref_height, unsigned char* ref_Y,

	//Camera data
	double* cam_K, double* cam_R, double* cam_t,
	int cam_width, int cam_height, unsigned char* cam_Y,

	//Output
	float* cost_cube, int zi, int window
) 
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= ref_width || y >= ref_height) {
		return;
	}

	double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

	double X_ref = (ref_K_inv[0] * x + ref_K_inv[1] * y + ref_K_inv[2]) * z;
	double Y_ref = (ref_K_inv[3] * x + ref_K_inv[4] * y + ref_K_inv[5]) * z;
	double Z_ref = (ref_K_inv[6] * x + ref_K_inv[7] * y + ref_K_inv[8]) * z;

	// 3D in ref camera coordinates to 3D world
	double X = ref_R_inv[0] * X_ref + ref_R_inv[1] * Y_ref + ref_R_inv[2] * Z_ref - ref_t_inv[0];
	double Y = ref_R_inv[3] * X_ref + ref_R_inv[4] * Y_ref + ref_R_inv[5] * Z_ref - ref_t_inv[1];
	double Z = ref_R_inv[6] * X_ref + ref_R_inv[7] * Y_ref + ref_R_inv[8] * Z_ref - ref_t_inv[2];

	// 3D world to projected camera 3D coordinates
	double X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_t[0];
	double Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_t[1];
	double Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	double x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
	double y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
	double z_proj = Z_proj;

	x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

	float cost = 0.0f;
	float cc = 0.0f;
	for (int k = -window / 2; k <= window / 2; k++)
	{
		for (int l = -window / 2; l <= window / 2; l++)
		{
			if (x + l < 0 || x + l >= ref_width)
				continue;
			if (y + k < 0 || y + k >= ref_height)
				continue;
			if (x_proj + l < 0 || x_proj + l >= cam_width)
				continue;
			if (y_proj + k < 0 || y_proj + k >= cam_height)
				continue;
			
			int ref_idx = (y + k) * ref_width + (x + l);
			int cam_idx = ((int)y_proj + k) * cam_width + ((int)x_proj + l);

			cost += fabs(ref_Y[ref_idx] - cam_Y[cam_idx]);

			cc += 1.0f;
		}
	}
	cost /= cc;

	int cost_idx = y * ref_width + x;

	// Store minimum cost (atomic to handle concurrent updates from different cameras)
	atomicMinf(&cost_cube[cost_idx], cost);
}

__device__ void atomicMinf(float* address, float val) {
	int* address_as_int = (int*)address;
	int old = *address_as_int;
	int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed,
			__float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
}

std::vector<cv::Mat> sweeping_plane_naive(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
	//function to call kernel
	//returns cost_cube to be used in main.cpp
}