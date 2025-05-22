#include "main.cuh"
#include <cstdio>
#include <cmath>
#include <chrono>

// Those functions are an example on how to call cuda functions from the main.cpp
__global__ void naive_kernel(
	//Reference data
	double* ref_K_inv, double* ref_R_inv, double* ref_t_inv,
	int ref_width, int ref_height, unsigned char* ref_Y,

	//Camera data
	double* cam_K, double* cam_R, double* cam_t,
	int cam_width, int cam_height, unsigned char* cam_Y,

	//Output
	float* cost_cube, int zi, int window,

	//Constants
	float ZNear, float ZFar, int ZPlanes
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

			cost += fabsf(ref_Y[ref_idx] - cam_Y[cam_idx]);

			cc += 1.0f;
		}
	}
	cost /= cc;

	int cost_idx = zi* ref_width * ref_height + y * ref_width + x;

    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}

__global__ void float_naive_kernel(
    // Reference data
    float* ref_K_inv, float* ref_R_inv, float* ref_t_inv,
    int ref_width, int ref_height, unsigned char* ref_Y,

    // Camera data
    float* cam_K, float* cam_R, float* cam_t,
    int cam_width, int cam_height, unsigned char* cam_Y,

    // Output
    float* cost_cube, int zi, int window,

    // Constants
    float ZNear, float ZFar, int ZPlanes, float z
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= ref_width || y >= ref_height) {
		return;
	}

	float X_ref = (ref_K_inv[0] * x + ref_K_inv[1] * y + ref_K_inv[2]) * z;
	float Y_ref = (ref_K_inv[3] * x + ref_K_inv[4] * y + ref_K_inv[5]) * z;
	float Z_ref = (ref_K_inv[6] * x + ref_K_inv[7] * y + ref_K_inv[8]) * z;

	// 3D in ref camera coordinates to 3D world
	float X = ref_R_inv[0] * X_ref + ref_R_inv[1] * Y_ref + ref_R_inv[2] * Z_ref - ref_t_inv[0];
	float Y = ref_R_inv[3] * X_ref + ref_R_inv[4] * Y_ref + ref_R_inv[5] * Z_ref - ref_t_inv[1];
	float Z = ref_R_inv[6] * X_ref + ref_R_inv[7] * Y_ref + ref_R_inv[8] * Z_ref - ref_t_inv[2];

	// 3D world to projected camera 3D coordinates
	float X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_t[0];
	float Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_t[1];
	float Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_t[2];

	// Projected camera 3D coordinates to projected camera 2D coordinates
	float x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
	float y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
	float z_proj = Z_proj;

	x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
	y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

	float cost = 0.0f;
	float cc = 0.0f;

    for (int k = -window / 2; k <= window / 2; k++) {
        for (int l = -window / 2; l <= window / 2; l++) {
            int ref_y = y + k;
            int ref_x = x + l;
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            if (ref_x >= 0 && ref_x < ref_width &&
                ref_y >= 0 && ref_y < ref_height &&
                cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int ref_idx = ref_y * ref_width + ref_x;
                int cam_idx = cam_y * cam_width + cam_x;

                cost += fabsf((float)ref_Y[ref_idx] - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }
	cost /= cc;

	int cost_idx = zi * ref_width * ref_height + y * ref_width + x;

	if (cost_cube[cost_idx] > cost) {
		cost_cube[cost_idx] = cost;
	}
}


__global__ void full_cam_kernel(
    // Reference data
    float* ref_K_inv, float* ref_R_inv, float* ref_t_inv,
    int ref_width, int ref_height, unsigned char* ref_Y,

    // Camera data
    float* cam_K, float* cam_R, float* cam_t,
    int cam_width, int cam_height, unsigned char* cam_Y,

    // Output
    float* cost_cube,int window,

    // Constants
    float ZNear, float ZFar, int ZPlanes
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z;


    if (x >= ref_width || y >= ref_height || zi>= ZPlanes) {
        return;
    }
    float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

    float X_ref = (ref_K_inv[0] * x + ref_K_inv[1] * y + ref_K_inv[2]) * z;
    float Y_ref = (ref_K_inv[3] * x + ref_K_inv[4] * y + ref_K_inv[5]) * z;
    float Z_ref = (ref_K_inv[6] * x + ref_K_inv[7] * y + ref_K_inv[8]) * z;

    // 3D in ref camera coordinates to 3D world
    float X = ref_R_inv[0] * X_ref + ref_R_inv[1] * Y_ref + ref_R_inv[2] * Z_ref - ref_t_inv[0];
    float Y = ref_R_inv[3] * X_ref + ref_R_inv[4] * Y_ref + ref_R_inv[5] * Z_ref - ref_t_inv[1];
    float Z = ref_R_inv[6] * X_ref + ref_R_inv[7] * Y_ref + ref_R_inv[8] * Z_ref - ref_t_inv[2];

    // 3D world to projected camera 3D coordinates
    float X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_t[0];
    float Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_t[1];
    float Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_t[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
    float y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
    float z_proj = Z_proj;

    x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -window / 2; k <= window / 2; k++) {
        for (int l = -window / 2; l <= window / 2; l++) {
            int ref_y = y + k;
            int ref_x = x + l;
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            if (ref_x >= 0 && ref_x < ref_width &&
                ref_y >= 0 && ref_y < ref_height &&
                cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int ref_idx = ref_y * ref_width + ref_x;
                int cam_idx = cam_y * cam_width + cam_x;

                cost += fabsf((float)ref_Y[ref_idx] - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }
    cost /= cc;

    int cost_idx = zi * ref_width * ref_height + y * ref_width + x;

    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}


__constant__ float d_ref_K_inv[9];
__constant__ float d_ref_R_inv[9];
__constant__ float d_ref_t_inv[3];
__constant__ float d_cam_K[9];
__constant__ float d_cam_R[9];
__constant__ float d_cam_t[3];


__global__ void constant_memory_kernel(
    // mettre les matrices en constant memory pour améliorer la latence
    // Reference data
    int ref_width, int ref_height, unsigned char* ref_Y,

    // Camera data
    int cam_width, int cam_height, unsigned char* cam_Y,

    // Output
    float* cost_cube, int window,

    // Constants
    float ZNear, float ZFar, int ZPlanes
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= ref_width || y >= ref_height || zi >= ZPlanes) {
        return;
    }
    float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

    float X_ref = (d_ref_K_inv[0] * x + d_ref_K_inv[1] * y + d_ref_K_inv[2]) * z;
    float Y_ref = (d_ref_K_inv[3] * x + d_ref_K_inv[4] * y + d_ref_K_inv[5]) * z;
    float Z_ref = (d_ref_K_inv[6] * x + d_ref_K_inv[7] * y + d_ref_K_inv[8]) * z;

    // 3D in ref camera coordinates to 3D world
    float X = d_ref_R_inv[0] * X_ref + d_ref_R_inv[1] * Y_ref + d_ref_R_inv[2] * Z_ref - d_ref_t_inv[0];
    float Y = d_ref_R_inv[3] * X_ref + d_ref_R_inv[4] * Y_ref + d_ref_R_inv[5] * Z_ref - d_ref_t_inv[1];
    float Z = d_ref_R_inv[6] * X_ref + d_ref_R_inv[7] * Y_ref + d_ref_R_inv[8] * Z_ref - d_ref_t_inv[2];

    // 3D world to projected camera 3D coordinates
    float X_proj = d_cam_R[0] * X + d_cam_R[1] * Y + d_cam_R[2] * Z - d_cam_t[0];
    float Y_proj = d_cam_R[3] * X + d_cam_R[4] * Y + d_cam_R[5] * Z - d_cam_t[1];
    float Z_proj = d_cam_R[6] * X + d_cam_R[7] * Y + d_cam_R[8] * Z - d_cam_t[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = (d_cam_K[0] * X_proj / Z_proj + d_cam_K[1] * Y_proj / Z_proj + d_cam_K[2]);
    float y_proj = (d_cam_K[3] * X_proj / Z_proj + d_cam_K[4] * Y_proj / Z_proj + d_cam_K[5]);
    
    //float z_proj = Z_proj;
    x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -window / 2; k <= window / 2; k++) {
        for (int l = -window / 2; l <= window / 2; l++) {
            int ref_y = y + k;
            int ref_x = x + l;
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            if (ref_x >= 0 && ref_x < ref_width &&
                ref_y >= 0 && ref_y < ref_height &&
                cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int ref_idx = ref_y * ref_width + ref_x;
                int cam_idx = cam_y * cam_width + cam_x;

                cost += fabsf((float)ref_Y[ref_idx] - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }
    cost /= cc;

    int cost_idx = zi * ref_width * ref_height + y * ref_width + x;

    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}


__global__ void float_matrix_kernel(
    // Reference data
    int ref_width, int ref_height, unsigned char* ref_Y,

    // Camera data
    float* cam_K, int cam_width, int cam_height, unsigned char* cam_Y,

    // Fused data
	float* R_cam_RK_ref, float* RT_cam_T_ref,

    // Output
    float* cost_cube, int zi, int window,

    // Constants
    float ZNear, float ZFar, int ZPlanes, float z
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= ref_width || y >= ref_height) {
        return;
    }

    // 3D world to projected camera 3D coordinates
    float X_proj = (R_cam_RK_ref[0] * x + R_cam_RK_ref[1] * y + R_cam_RK_ref[2]) * z - RT_cam_T_ref[0];
    float Y_proj = (R_cam_RK_ref[3] * x + R_cam_RK_ref[4] * y + R_cam_RK_ref[5]) * z - RT_cam_T_ref[1];
    float Z_proj = (R_cam_RK_ref[6] * x + R_cam_RK_ref[7] * y + R_cam_RK_ref[8]) * z - RT_cam_T_ref[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = (cam_K[0] * X_proj / Z_proj + cam_K[1] * Y_proj / Z_proj + cam_K[2]);
    float y_proj = (cam_K[3] * X_proj / Z_proj + cam_K[4] * Y_proj / Z_proj + cam_K[5]);
    float z_proj = Z_proj;

    x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -window / 2; k <= window / 2; k++) {
        for (int l = -window / 2; l <= window / 2; l++) {
            int ref_y = y + k;
            int ref_x = x + l;
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            if (ref_x >= 0 && ref_x < ref_width &&
                ref_y >= 0 && ref_y < ref_height &&
                cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int ref_idx = ref_y * ref_width + ref_x;
                int cam_idx = cam_y * cam_width + cam_x;

                cost += fabsf((float)ref_Y[ref_idx] - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }
    cost /= cc;

    int cost_idx = zi * ref_width * ref_height + y * ref_width + x;

    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}



__constant__ float  d_R_cam_RK_ref[9];
__constant__ float  d_RT_cam_T_ref[3];
__global__ void constant_memory_matrix_kernel(
    // Reference data
    int ref_width, int ref_height, unsigned char* ref_Y,

    // Camera data
    int cam_width, int cam_height, unsigned char* cam_Y,

    // Output
    float* cost_cube, int window,

    // Constants
    float ZNear, float ZFar, int ZPlanes
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z;


    if (x >= ref_width || y >= ref_height || zi >= ZPlanes) {
        return;
    }
    float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

    // 3D world to projected camera 3D coordinates
    float X_proj = (d_R_cam_RK_ref[0] * x + d_R_cam_RK_ref[1] * y + d_R_cam_RK_ref[2]) * z - d_RT_cam_T_ref[0];
    float Y_proj = (d_R_cam_RK_ref[3] * x + d_R_cam_RK_ref[4] * y + d_R_cam_RK_ref[5]) * z - d_RT_cam_T_ref[1];
    float Z_proj = (d_R_cam_RK_ref[6] * x + d_R_cam_RK_ref[7] * y + d_R_cam_RK_ref[8]) * z - d_RT_cam_T_ref[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = (d_cam_K[0] * X_proj / Z_proj + d_cam_K[1] * Y_proj / Z_proj + d_cam_K[2]);
    float y_proj = (d_cam_K[3] * X_proj / Z_proj + d_cam_K[4] * Y_proj / Z_proj + d_cam_K[5]);
    float z_proj = Z_proj;

    x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -window / 2; k <= window / 2; k++) {
        for (int l = -window / 2; l <= window / 2; l++) {
            int ref_y = y + k;
            int ref_x = x + l;
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            if (ref_x >= 0 && ref_x < ref_width &&
                ref_y >= 0 && ref_y < ref_height &&
                cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int ref_idx = ref_y * ref_width + ref_x;
                int cam_idx = cam_y * cam_width + cam_x;

                cost += fabsf((float)ref_Y[ref_idx] - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }
    cost /= cc;

    int cost_idx = zi * ref_width * ref_height + y * ref_width + x;

    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}

__global__ void grid3d_kernel(
    // Reference data
    int ref_width, int ref_height, unsigned char* ref_Y,

    // Camera data
    int cam_width, int cam_height, unsigned char* cam_Y,

    // Output
    float* cost_cube, int window,

    // Constants
    float ZNear, float ZFar, int ZPlanes
)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z;

    if (x >= ref_width || y >= ref_height || zi >= ZPlanes) {
        return;
    }
    float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

    // 3D world to projected camera 3D coordinates
    float X_proj = (d_R_cam_RK_ref[0] * x + d_R_cam_RK_ref[1] * y + d_R_cam_RK_ref[2]) * z - d_RT_cam_T_ref[0];
    float Y_proj = (d_R_cam_RK_ref[3] * x + d_R_cam_RK_ref[4] * y + d_R_cam_RK_ref[5]) * z - d_RT_cam_T_ref[1];
    float Z_proj = (d_R_cam_RK_ref[6] * x + d_R_cam_RK_ref[7] * y + d_R_cam_RK_ref[8]) * z - d_RT_cam_T_ref[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = (d_cam_K[0] * X_proj / Z_proj + d_cam_K[1] * Y_proj / Z_proj + d_cam_K[2]);
    float y_proj = (d_cam_K[3] * X_proj / Z_proj + d_cam_K[4] * Y_proj / Z_proj + d_cam_K[5]);
    float z_proj = Z_proj;

    x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -window / 2; k <= window / 2; k++) {
        for (int l = -window / 2; l <= window / 2; l++) {
            int ref_y = y + k;
            int ref_x = x + l;
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            if (ref_x >= 0 && ref_x < ref_width &&
                ref_y >= 0 && ref_y < ref_height &&
                cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int ref_idx = ref_y * ref_width + ref_x;
                int cam_idx = cam_y * cam_width + cam_x;

                cost += fabsf((float)ref_Y[ref_idx] - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }
    cost /= cc;

    int cost_idx = zi * ref_width * ref_height + y * ref_width + x;

    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}

__global__ void grid3d_shared_ref_kernel(
    // Reference data
    int ref_width, int ref_height, unsigned char* ref_Y,
    // Camera data
    int cam_width, int cam_height, unsigned char* cam_Y,
    // Output
    float* cost_cube, int window,
    // Constants
    float ZNear, float ZFar, int ZPlanes
)
{
    // 2D block, 3D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z; // Each z-plane handled by a separate grid layer

    // Calculate shared memory size with padding for window
    int padding = window / 2;
    int smem_width = blockDim.x + 2 * padding;
    int smem_height = blockDim.y + 2 * padding;

    // Declare shared memory for reference image with padding
    extern __shared__ unsigned char s_ref[];

    // Local thread indices (for shared memory access)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate block start coordinates in global memory
    int block_start_x = blockIdx.x * blockDim.x - padding;
    int block_start_y = blockIdx.y * blockDim.y - padding;

    // Load primary position (every thread loads its own position)
    int s_idx = (ty + padding) * smem_width + (tx + padding);
    if (x < ref_width && y < ref_height) {
        s_ref[s_idx] = ref_Y[y * ref_width + x];
    }
    else {
        s_ref[s_idx] = 0; // Default value for out of bounds
    }

    // Load top edge (first row of threads)
    if (ty == 0) {
        for (int i = 0; i < padding; i++) {
            int load_y = block_start_y + i;
            // Clamp to valid reference image coordinates
            load_y = max(0, min(ref_height - 1, load_y));
            int s_y_idx = i * smem_width + (tx + padding);
            int g_y_idx = load_y * ref_width + x;

            if (x < ref_width) {
                s_ref[s_y_idx] = ref_Y[g_y_idx];
            }
            else {
                s_ref[s_y_idx] = 0;
            }
        }
    }

    // Load bottom edge (last row of threads)
    if (ty == blockDim.y - 1 || y == ref_height - 1) {
        for (int i = 1; i <= padding; i++) {
            int load_y = y + i;
            // Clamp to valid reference image coordinates
            load_y = max(0, min(ref_height - 1, load_y));
            int s_y_idx = (ty + padding + i) * smem_width + (tx + padding);
            int g_y_idx = load_y * ref_width + x;

            if (x < ref_width) {
                s_ref[s_y_idx] = ref_Y[g_y_idx];
            }
            else {
                s_ref[s_y_idx] = 0;
            }
        }
    }

    // Load left edge (first column of threads)
    if (tx == 0) {
        for (int i = 0; i < padding; i++) {
            int load_x = block_start_x + i;
            // Clamp to valid reference image coordinates
            load_x = max(0, min(ref_width - 1, load_x));
            int s_x_idx = (ty + padding) * smem_width + i;
            int g_x_idx = y * ref_width + load_x;

            if (y < ref_height) {
                s_ref[s_x_idx] = ref_Y[g_x_idx];
            }
            else {
                s_ref[s_x_idx] = 0;
            }
        }
    }

    // Load right edge (last column of threads)
    if (tx == blockDim.x - 1 || x == ref_width - 1) {
        for (int i = 1; i <= padding; i++) {
            int load_x = x + i;
            // Clamp to valid reference image coordinates
            load_x = max(0, min(ref_width - 1, load_x));
            int s_x_idx = (ty + padding) * smem_width + (tx + padding + i);
            int g_x_idx = y * ref_width + load_x;

            if (y < ref_height) {
                s_ref[s_x_idx] = ref_Y[g_x_idx];
            }
            else {
                s_ref[s_x_idx] = 0;
            }
        }
    }

    // Load top-left corner
    if (tx == 0 && ty == 0) {
        for (int j = 0; j < padding; j++) {
            for (int i = 0; i < padding; i++) {
                int load_y = block_start_y + j;
                int load_x = block_start_x + i;
                // Clamp to valid reference image coordinates
                load_y = max(0, min(ref_height - 1, load_y));
                load_x = max(0, min(ref_width - 1, load_x));

                int s_corner_idx = j * smem_width + i;
                int g_corner_idx = load_y * ref_width + load_x;
                s_ref[s_corner_idx] = ref_Y[g_corner_idx];
            }
        }
    }

    // Load top-right corner
    if (tx == blockDim.x - 1 && ty == 0) {
        for (int j = 0; j < padding; j++) {
            for (int i = 1; i <= padding; i++) {
                int load_y = block_start_y + j;
                int load_x = x + i;
                // Clamp to valid reference image coordinates
                load_y = max(0, min(ref_height - 1, load_y));
                load_x = max(0, min(ref_width - 1, load_x));

                int s_corner_idx = j * smem_width + (tx + padding + i);
                int g_corner_idx = load_y * ref_width + load_x;
                s_ref[s_corner_idx] = ref_Y[g_corner_idx];
            }
        }
    }

    // Load bottom-left corner
    if (tx == 0 && (ty == blockDim.y - 1 || y == ref_height - 1)) {
        for (int j = 1; j <= padding; j++) {
            for (int i = 0; i < padding; i++) {
                int load_y = y + j;
                int load_x = block_start_x + i;
                // Clamp to valid reference image coordinates
                load_y = max(0, min(ref_height - 1, load_y));
                load_x = max(0, min(ref_width - 1, load_x));

                int s_corner_idx = (ty + padding + j) * smem_width + i;
                int g_corner_idx = load_y * ref_width + load_x;
                s_ref[s_corner_idx] = ref_Y[g_corner_idx];
            }
        }
    }

    // Load bottom-right corner
    if ((tx == blockDim.x - 1 || x == ref_width - 1) &&
        (ty == blockDim.y - 1 || y == ref_height - 1)) {
        for (int j = 1; j <= padding; j++) {
            for (int i = 1; i <= padding; i++) {
                int load_y = y + j;
                int load_x = x + i;
                // Clamp to valid reference image coordinates
                load_y = max(0, min(ref_height - 1, load_y));
                load_x = max(0, min(ref_width - 1, load_x));

                int s_corner_idx = (ty + padding + j) * smem_width + (tx + padding + i);
                int g_corner_idx = load_y * ref_width + load_x;
                s_ref[s_corner_idx] = ref_Y[g_corner_idx];
            }
        }
    }

    // Ensure all threads have loaded data into shared memory
    __syncthreads();

    if (x >= ref_width || y >= ref_height || zi >= ZPlanes) {
        return;
    }

    // Calculate depth for this plane
    float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

    // 3D world to projected camera 3D coordinates
    float X_proj = (d_R_cam_RK_ref[0] * x + d_R_cam_RK_ref[1] * y + d_R_cam_RK_ref[2]) * z - d_RT_cam_T_ref[0];
    float Y_proj = (d_R_cam_RK_ref[3] * x + d_R_cam_RK_ref[4] * y + d_R_cam_RK_ref[5]) * z - d_RT_cam_T_ref[1];
    float Z_proj = (d_R_cam_RK_ref[6] * x + d_R_cam_RK_ref[7] * y + d_R_cam_RK_ref[8]) * z - d_RT_cam_T_ref[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = (d_cam_K[0] * X_proj / Z_proj + d_cam_K[1] * Y_proj / Z_proj + d_cam_K[2]);
    float y_proj = (d_cam_K[3] * X_proj / Z_proj + d_cam_K[4] * Y_proj / Z_proj + d_cam_K[5]);

    // Check if projection is valid and round to nearest pixel
    x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

    // Calculate cost using shared memory for reference image
    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -padding; k <= padding; k++) {
        for (int l = -padding; l <= padding; l++) {
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            // Access reference pixel from shared memory
            int s_ref_y = ty + padding + k;
            int s_ref_x = tx + padding + l;

            if (cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int cam_idx = cam_y * cam_width + cam_x;
                unsigned char ref_val = s_ref[s_ref_y * smem_width + s_ref_x];
                cost += fabsf((float)ref_val - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }

    // Average the cost
    if (cc > 0) {
        cost /= cc;
    }

    // Update cost cube if new cost is lower
    int cost_idx = zi * ref_width * ref_height + y * ref_width + x;
    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}

__constant__ float d_planes[ZPlanes];

__global__ void grid3d_shared_ref_2_kernel(
    // Reference data
    int ref_width, int ref_height, unsigned char* ref_Y,
    // Camera data
    int cam_width, int cam_height, unsigned char* cam_Y,
    // Output
    float* cost_cube, int window,
    // Constants
    float ZNear, float ZFar, int ZPlanes
)
{
    // 2D block, 3D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z; // Each z-plane handled by a separate grid layer

    if (x >= ref_width || y >= ref_height || zi >= ZPlanes) {
        return;
    }

    // Calculate shared memory size with padding for window
    int padding = window / 2;
    int smem_width = blockDim.x + 2 * padding;
    int smem_height = blockDim.y + 2 * padding;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Declare shared memory for reference image with padding
    extern __shared__ unsigned char s_ref[];

    int pixels_per_thread = (smem_width * smem_height + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);
    int thread_id = ty * blockDim.x + tx;

    for (int p = 0; p < pixels_per_thread; p++) {
        int pixel_id = thread_id * pixels_per_thread + p;
        if (pixel_id < smem_width * smem_height) {
            int s_y = pixel_id / smem_width;
            int s_x = pixel_id % smem_width;

            int g_x = blockIdx.x * blockDim.x + s_x - padding;
            int g_y = blockIdx.y * blockDim.y + s_y - padding;

            // Clamp coordinates
            g_x = max(0, min(ref_width - 1, g_x));
            g_y = max(0, min(ref_height - 1, g_y));

            s_ref[pixel_id] = ref_Y[g_y * ref_width + g_x];
        }
    }



    // Ensure all threads have loaded data into shared memory
    __syncthreads();



    // Calculate depth for this plane
    float z = d_planes[zi]; //= ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

    // 3D world to projected camera 3D coordinates
    float X_proj = (d_R_cam_RK_ref[0] * x + d_R_cam_RK_ref[1] * y + d_R_cam_RK_ref[2]) * z - d_RT_cam_T_ref[0];
    float Y_proj = (d_R_cam_RK_ref[3] * x + d_R_cam_RK_ref[4] * y + d_R_cam_RK_ref[5]) * z - d_RT_cam_T_ref[1];
    float Z_proj = (d_R_cam_RK_ref[6] * x + d_R_cam_RK_ref[7] * y + d_R_cam_RK_ref[8]) * z - d_RT_cam_T_ref[2];

    // Projected camera 3D coordinates to projected camera 2D coordinates
    float x_proj = (d_cam_K[0] * X_proj / Z_proj + d_cam_K[1] * Y_proj / Z_proj + d_cam_K[2]);
    float y_proj = (d_cam_K[3] * X_proj / Z_proj + d_cam_K[4] * Y_proj / Z_proj + d_cam_K[5]);

    // Check if projection is valid and round to nearest pixel
    x_proj = x_proj < 0 || x_proj >= cam_width ? 0 : roundf(x_proj);
    y_proj = y_proj < 0 || y_proj >= cam_height ? 0 : roundf(y_proj);

    // Calculate cost using shared memory for reference image
    float cost = 0.0f;
    float cc = 0.0f;

    for (int k = -padding; k <= padding; k++) {
        for (int l = -padding; l <= padding; l++) {
            int cam_y = y_proj + k;
            int cam_x = x_proj + l;

            // Access reference pixel from shared memory
            int s_ref_y = ty + padding + k;
            int s_ref_x = tx + padding + l;

            if (cam_x >= 0 && cam_x < cam_width &&
                cam_y >= 0 && cam_y < cam_height) {

                int cam_idx = cam_y * cam_width + cam_x;
                unsigned char ref_val = s_ref[s_ref_y * smem_width + s_ref_x];
                cost += fabsf((float)ref_val - (float)cam_Y[cam_idx]);
                cc += 1.0f;
            }
        }
    }

    // Average the cost
    if (cc > 0) {
        cost /= cc;
    }

    // Update cost cube if new cost is lower
    int cost_idx = zi * ref_width * ref_height + y * ref_width + x;
    if (cost_cube[cost_idx] > cost) {
        cost_cube[cost_idx] = cost;
    }
}

std::vector<cv::Mat> sweeping_plane_naive(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
	//function to call kernel
	//returns cost_cube to be used in main.cpp
	int width = ref.width;
	int height = ref.height;
	int total_size = width * height;

	std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
	float* d_cost_cube;
	size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
	cudaMalloc((void**)&d_cost_cube, cost_cube_size);
	cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

	double* d_ref_K_inv, * d_ref_R_inv, * d_ref_t_inv;
	cudaMalloc((void**)&d_ref_K_inv, 9 * sizeof(double));
	cudaMalloc((void**)&d_ref_R_inv, 9 * sizeof(double));
	cudaMalloc((void**)&d_ref_t_inv, 3 * sizeof(double));

	cudaMemcpy(d_ref_K_inv, ref.p.K_inv.data(), 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_R_inv, ref.p.R_inv.data(), 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_t_inv, ref.p.t_inv.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);

	unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
	std::cout << "ref stride: " << ref_stride << std::endl;
	std::cout << "ref width: " << width << std::endl;
	std::cout << "ref height: " << height << std::endl;
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

	for (auto& cam : cam_vector){
		if (cam.name == ref.name){
			continue;
		}

		std::cout << "Cam: " << cam.name << std::endl;

		double* d_cam_K, * d_cam_R, * d_cam_t;
		cudaMalloc((void**)&d_cam_K, 9 * sizeof(double));
		cudaMalloc((void**)&d_cam_R, 9 * sizeof(double));
		cudaMalloc((void**)&d_cam_t, 3 * sizeof(double));

		cudaMemcpy(d_cam_K, cam.p.K.data(), 9 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cam_R, cam.p.R.data(), 9 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cam_t, cam.p.t.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);

		unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
		std::cout << "cam stride: " << cam_stride << std::endl;
		std::cout << "cam width: " << cam.width << std::endl;
		std::cout << "cam height: " << cam.height << std::endl;
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
		dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
        
		for (int zi = 0; zi < ZPlanes; zi++) {
            std::cout << "Plane " << zi << std::endl;
            
            // Launch kernel
            naive_kernel<<<gridDim, blockDim>>>(
                d_ref_K_inv, d_ref_R_inv, d_ref_t_inv,
                 width, height, d_ref_Y,
                d_cam_K, d_cam_R, d_cam_t,
                cam.width, cam.height, d_cam_Y,
                d_cost_cube, zi, window,
				ZNear, ZFar, ZPlanes
            );

			// Check for errors
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            }
		}
		
		cudaFree(d_cam_K);
        cudaFree(d_cam_R);
        cudaFree(d_cam_t);
        cudaFree(d_cam_Y); 
	}

	cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

	std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        // Copy the appropriate slice of the cost_cube_data into the cv::Mat
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
				int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
		//printf("%d, ", result[i]);
    }
	cudaFree(d_ref_K_inv);
    cudaFree(d_ref_R_inv);
    cudaFree(d_ref_t_inv);
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

	return result;
}

std::vector<cv::Mat> sweeping_plane_float_naive(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }

    // Allocate and copy reference camera matrices
    float* d_ref_K_inv, * d_ref_R_inv, * d_ref_t_inv;
    cudaMalloc((void**)&d_ref_K_inv, 9 * sizeof(float));
    cudaMalloc((void**)&d_ref_R_inv, 9 * sizeof(float));
    cudaMalloc((void**)&d_ref_t_inv, 3 * sizeof(float));

    cudaMemcpy(d_ref_K_inv, ref_K_inv_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_R_inv, ref_R_inv_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_t_inv, ref_t_inv_float.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;
        }

        std::cout << "Cam: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Allocate and copy camera matrices
        float* d_cam_K, * d_cam_R, * d_cam_t;
        cudaMalloc((void**)&d_cam_K, 9 * sizeof(float));
        cudaMalloc((void**)&d_cam_R, 9 * sizeof(float));
        cudaMalloc((void**)&d_cam_t, 3 * sizeof(float));

        cudaMemcpy(d_cam_K, cam_K_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cam_R, cam_R_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cam_t, cam_t_float.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set kernel launch parameters
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Process each depth plane
        for (int zi = 0; zi < ZPlanes; zi++) {
            std::cout << "Plane " << zi << std::endl;
            float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

            // Launch kernel with float parameters
            float_naive_kernel << <gridDim, blockDim >> > (
                d_ref_K_inv, d_ref_R_inv, d_ref_t_inv,
                width, height, d_ref_Y,
                d_cam_K, d_cam_R, d_cam_t,
                cam.width, cam.height, d_cam_Y,
                d_cost_cube, zi, window,
                static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes, z
                );

            // Check for errors
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            }
        }

        // Free camera resources
        cudaFree(d_cam_K);
        cudaFree(d_cam_R);
        cudaFree(d_cam_t);
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_K_inv);
    cudaFree(d_ref_R_inv);
    cudaFree(d_ref_t_inv);
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

std::vector<cv::Mat> sweeping_plane_full_cam(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }

    // Allocate and copy reference camera matrices
    float* d_ref_K_inv, * d_ref_R_inv, * d_ref_t_inv;
    cudaMalloc((void**)&d_ref_K_inv, 9 * sizeof(float));
    cudaMalloc((void**)&d_ref_R_inv, 9 * sizeof(float));
    cudaMalloc((void**)&d_ref_t_inv, 3 * sizeof(float));

    cudaMemcpy(d_ref_K_inv, ref_K_inv_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_R_inv, ref_R_inv_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_t_inv, ref_t_inv_float.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    std::cout << "ref stride: " << ref_stride << std::endl;
    std::cout << "ref width: " << width << std::endl;
    std::cout << "ref height: " << height << std::endl;
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;
        }

        std::cout << "Cam: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Allocate and copy camera matrices
        float* d_cam_K, * d_cam_R, * d_cam_t;
        cudaMalloc((void**)&d_cam_K, 9 * sizeof(float));
        cudaMalloc((void**)&d_cam_R, 9 * sizeof(float));
        cudaMalloc((void**)&d_cam_t, 3 * sizeof(float));

        cudaMemcpy(d_cam_K, cam_K_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cam_R, cam_R_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cam_t, cam_t_float.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set kernel launch parameters
        dim3 blockDim(16, 16, 4);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, (ZPlanes + blockDim.z - 1) / blockDim.z);

        // Launch kernel with float parameters
        full_cam_kernel << <gridDim, blockDim >> > (
            d_ref_K_inv, d_ref_R_inv, d_ref_t_inv,
            width, height, d_ref_Y,
            d_cam_K, d_cam_R, d_cam_t,
            cam.width, cam.height, d_cam_Y,
            d_cost_cube, window,
            static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes
            );

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            
        }

        // Free camera resources
        cudaFree(d_cam_K);
        cudaFree(d_cam_R);
        cudaFree(d_cam_t);
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_K_inv);
    cudaFree(d_ref_R_inv);
    cudaFree(d_ref_t_inv);
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

std::vector<cv::Mat> sweeping_plane_constant_mem(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }

    // Allocate and copy reference camera matrices
    cudaMemcpyToSymbol(d_ref_K_inv, ref_K_inv_float.data(), 9 * sizeof(float));
    cudaMemcpyToSymbol(d_ref_R_inv, ref_R_inv_float.data(), 9 * sizeof(float));
    cudaMemcpyToSymbol(d_ref_t_inv, ref_t_inv_float.data(), 3 * sizeof(float));

    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;
        }

        std::cout << "Cam: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }
        cudaMemcpyToSymbol(d_cam_K, cam_K_float.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_cam_R, cam_R_float.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_cam_t, cam_t_float.data(), 3 * sizeof(float));

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set kernel launch parameters
        dim3 blockDim(16, 16, 4);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, (ZPlanes + blockDim.z - 1) / blockDim.z);

        // Launch kernel with float parameters
        constant_memory_kernel << <gridDim, blockDim >> > (
            width, height, d_ref_Y,
            cam.width, cam.height, d_cam_Y,
            d_cost_cube, window,
            static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes
            );

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

        }
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

std::vector<cv::Mat> sweeping_plane_reduced_maxtrix(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }


    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;
        }

        std::cout << "Cam: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Matrix multiplication cam_R* ref_R_inv * ref_K_inv
        std::vector<float> temporary(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                   sum += cam_R_float[i * 3 + k] * ref_R_inv_float[k * 3 + j];
                }
				temporary[i * 3 + j] = sum;
            }
        }

        std::vector<float> R_cam_RT_ref(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += temporary[i * 3 + k] * ref_K_inv_float[k * 3 + j];
                }
                R_cam_RT_ref[i * 3 + j] = sum;
            }
        }

        std::vector<float> RT_cam_T_ref(3);
        for (int i = 0; i < 3; i++) {
            float sum = 0.0;
            for (int j = 0; j < 3; j++) {
                sum += cam_R_float[i * 3 + j] * cam_t_float[j];
            }
			RT_cam_T_ref[i] = sum + ref_t_inv_float[i];
        }
        
        // Allocate and copy camera matrices
        float* d_R_cam_RK_ref, * d_RT_cam_T_ref, * d_cam_K;
        cudaMalloc((void**)&d_R_cam_RK_ref, 9 * sizeof(float));
        cudaMalloc((void**)&d_cam_K, 9 * sizeof(float));
        cudaMalloc((void**)&d_RT_cam_T_ref, 3 * sizeof(float));

        cudaMemcpy(d_R_cam_RK_ref, R_cam_RT_ref.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cam_K, cam_K_float.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_RT_cam_T_ref, RT_cam_T_ref.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set kernel launch parameters
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Process each depth plane
        for (int zi = 0; zi < ZPlanes; zi++) {
            std::cout << "Plane " << zi << std::endl;
            float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));

            // Launch kernel with float parameters
            float_matrix_kernel << <gridDim, blockDim >> > (
                width, height, d_ref_Y,
                d_cam_K, cam.width, cam.height, d_cam_Y,
                d_R_cam_RK_ref, d_RT_cam_T_ref,
                d_cost_cube, zi, window,
                static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes, z
                );

            // Check for errors
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            }
        }

        // Free camera resources
        cudaFree(d_cam_K);
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

std::vector<cv::Mat> sweeping_plane_constant_mem_matrix(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }
    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);
    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;
        }
        std::cout << "Cam: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Matrix multiplication cam_R* ref_R_inv * ref_K_inv
        std::vector<float> temporary(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += cam_R_float[i * 3 + k] * ref_R_inv_float[k * 3 + j];
                }
                temporary[i * 3 + j] = sum;
            }
        }

        std::vector<float> R_cam_RK_ref(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += temporary[i * 3 + k] * ref_K_inv_float[k * 3 + j];
                }
                R_cam_RK_ref[i * 3 + j] = sum;
            }
        }

        std::vector<float> RT_cam_T_ref(3);
        for (int i = 0; i < 3; i++) {
            float sum = 0.0;
            for (int j = 0; j < 3; j++) {
                sum += cam_R_float[i * 3 + j] * cam_t_float[j];
            }
            RT_cam_T_ref[i] = sum + ref_t_inv_float[i];
        }

        // Allocate and copy matrices
        cudaMemcpyToSymbol(d_cam_K, cam_K_float.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_R_cam_RK_ref, R_cam_RK_ref.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_RT_cam_T_ref, RT_cam_T_ref.data(), 3 * sizeof(float));

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set kernel launch parameters
        dim3 blockDim(16, 16, 4);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, (ZPlanes + blockDim.z - 1) / blockDim.z);

        // Launch kernel with float parameters
        constant_memory_matrix_kernel << <gridDim, blockDim >> > (
            width, height, d_ref_Y,
            cam.width, cam.height, d_cam_Y,
            d_cost_cube, window,
            static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes
            );

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

        }

        // Free camera resources
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

std::vector<cv::Mat> sweeping_plane_grid3d(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }
    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);
    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;
        }
        std::cout << "Cam: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Matrix multiplication cam_R* ref_R_inv * ref_K_inv
        std::vector<float> temporary(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += cam_R_float[i * 3 + k] * ref_R_inv_float[k * 3 + j];
                }
                temporary[i * 3 + j] = sum;
            }
        }

        std::vector<float> R_cam_RK_ref(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += temporary[i * 3 + k] * ref_K_inv_float[k * 3 + j];
                }
                R_cam_RK_ref[i * 3 + j] = sum;
            }
        }

        std::vector<float> RT_cam_T_ref(3);
        for (int i = 0; i < 3; i++) {
            float sum = 0.0;
            for (int j = 0; j < 3; j++) {
                sum += cam_R_float[i * 3 + j] * cam_t_float[j];
            }
            RT_cam_T_ref[i] = sum + ref_t_inv_float[i];
        }

        // Allocate and copy matrices
        cudaMemcpyToSymbol(d_cam_K, cam_K_float.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_R_cam_RK_ref, R_cam_RK_ref.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_RT_cam_T_ref, RT_cam_T_ref.data(), 3 * sizeof(float));

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set kernel launch parameters
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, ZPlanes);

        // Launch kernel with float parameters
        grid3d_kernel << <gridDim, blockDim >> > (
            width, height, d_ref_Y,
            cam.width, cam.height, d_cam_Y,
            d_cost_cube, window,
            static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes
            );

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

        }
        cudaDeviceSynchronize();

        // Free camera resources
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

std::vector<cv::Mat> sweeping_plane_grid3d_shared_ref(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }

    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;  // Skip reference camera
        }
        std::cout << "Processing camera: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Matrix multiplication cam_R * ref_R_inv * ref_K_inv
        std::vector<float> temporary(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += cam_R_float[i * 3 + k] * ref_R_inv_float[k * 3 + j];
                }
                temporary[i * 3 + j] = sum;
            }
        }

        std::vector<float> R_cam_RK_ref(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += temporary[i * 3 + k] * ref_K_inv_float[k * 3 + j];
                }
                R_cam_RK_ref[i * 3 + j] = sum;
            }
        }

        std::vector<float> RT_cam_T_ref(3);
        for (int i = 0; i < 3; i++) {
            float sum = 0.0;
            for (int j = 0; j < 3; j++) {
                sum += cam_R_float[i * 3 + j] * cam_t_float[j];
            }
            RT_cam_T_ref[i] = sum + ref_t_inv_float[i];
        }

        // Allocate and copy matrices to constant memory
        cudaMemcpyToSymbol(d_cam_K, cam_K_float.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_R_cam_RK_ref, R_cam_RK_ref.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_RT_cam_T_ref, RT_cam_T_ref.data(), 3 * sizeof(float));

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set 2D block dimensions
        dim3 blockDim(16, 16);

        // Calculate padding for shared memory
        int padding = window / 2;

        // Calculate shared memory size for reference image with padding
        int smem_width = blockDim.x + 2 * padding;
        int smem_height = blockDim.y + 2 * padding;
        size_t smem_size = smem_width * smem_height * sizeof(unsigned char);

        // Set 3D grid dimensions
        dim3 gridDim(
            (width + blockDim.x - 1) / blockDim.x,
            (height + blockDim.y - 1) / blockDim.y,
            ZPlanes  // One grid layer per Z plane
        );

        // Launch kernel with 3D grid, 2D blocks, and shared memory
        grid3d_shared_ref_kernel << <gridDim, blockDim, smem_size >> > (
            width, height, d_ref_Y,
            cam.width, cam.height, d_cam_Y,
            d_cost_cube, window,
            static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes
            );

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }

        // Synchronize to ensure kernel execution completes
        cudaDeviceSynchronize();

        // Free camera resources
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

std::vector<cv::Mat> sweeping_plane_grid3d_shared_ref_2(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes, 255.0f);
    float* d_cost_cube;
    size_t cost_cube_size = total_size * ZPlanes * sizeof(float);
    cudaMalloc((void**)&d_cost_cube, cost_cube_size);
    cudaMemcpy(d_cost_cube, cost_cube_data.data(), cost_cube_size, cudaMemcpyHostToDevice);

    std::vector<float> h_planes(ZPlanes);
    for (int zi = 0; zi < ZPlanes; ++zi) {
        h_planes[zi] = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
    }
    cudaMemcpyToSymbol(d_planes, h_planes.data(), ZPlanes * sizeof(float));

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }

    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;  // Skip reference camera
        }
        std::cout << "Processing camera: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Matrix multiplication cam_R * ref_R_inv * ref_K_inv
        std::vector<float> temporary(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += cam_R_float[i * 3 + k] * ref_R_inv_float[k * 3 + j];
                }
                temporary[i * 3 + j] = sum;
            }
        }

        std::vector<float> R_cam_RK_ref(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += temporary[i * 3 + k] * ref_K_inv_float[k * 3 + j];
                }
                R_cam_RK_ref[i * 3 + j] = sum;
            }
        }

        std::vector<float> RT_cam_T_ref(3);
        for (int i = 0; i < 3; i++) {
            float sum = 0.0;
            for (int j = 0; j < 3; j++) {
                sum += cam_R_float[i * 3 + j] * cam_t_float[j];
            }
            RT_cam_T_ref[i] = sum + ref_t_inv_float[i];
        }

        // Allocate and copy matrices to constant memory
        cudaMemcpyToSymbol(d_cam_K, cam_K_float.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_R_cam_RK_ref, R_cam_RK_ref.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_RT_cam_T_ref, RT_cam_T_ref.data(), 3 * sizeof(float));

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set 2D block dimensions
        dim3 blockDim(32, 32);
        //dim3 blockDim(16, 16);

        // Calculate padding for shared memory
        int padding = window / 2;

        // Calculate shared memory size for reference image with padding
        int smem_width = blockDim.x + 2 * padding;
        int smem_height = blockDim.y + 2 * padding;
        size_t smem_size = smem_width * smem_height * sizeof(unsigned char);

        // Set 3D grid dimensions
        dim3 gridDim(
            (width + blockDim.x - 1) / blockDim.x,
            (height + blockDim.y - 1) / blockDim.y,
            ZPlanes  // One grid layer per Z plane
        );

        // Launch kernel with 3D grid, 2D blocks, and shared memory
        auto start = std::chrono::high_resolution_clock::now();

        grid3d_shared_ref_2_kernel << <gridDim, blockDim, smem_size >> > (
            width, height, d_ref_Y,
            cam.width, cam.height, d_cam_Y,
            d_cost_cube, window,
            static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes
            );

        auto end = std::chrono::high_resolution_clock::now();
        // Print time duration of algorithm
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() * 1000 << " ms" << std::endl;

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }

        // Synchronize to ensure kernel execution completes
        cudaDeviceSynchronize();

        // Free camera resources
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = i * width * height + y * width + x;
                result[i].at<float>(y, x) = cost_cube_data[index];
            }
        }
    }

    // Free reference resources
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}

__global__ void init_cost_cube_kernel(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

std::vector<cv::Mat> sweeping_plane_final(cam const& ref, std::vector<cam> const& cam_vector, int window = 3) {
    int width = ref.width;
    int height = ref.height;
    int total_size = width * height;

    // Initialize cost cube with max values
    std::vector<float> cost_cube_data(total_size * ZPlanes);
    float* d_cost_cube;
    int total_elements = total_size * ZPlanes;
    size_t cost_cube_size = total_elements * sizeof(float);
    cudaMalloc(&d_cost_cube, cost_cube_size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    init_cost_cube_kernel << <blocksPerGrid, threadsPerBlock >> > (d_cost_cube, 255.0f, total_elements);
    cudaDeviceSynchronize();

    std::vector<float> h_planes(ZPlanes);
    for (int zi = 0; zi < ZPlanes; ++zi) {
        h_planes[zi] = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
    }
    cudaMemcpyToSymbol(d_planes, h_planes.data(), ZPlanes * sizeof(float));

    // Convert reference camera matrices to float
    std::vector<float> ref_K_inv_float(9), ref_R_inv_float(9), ref_t_inv_float(3);
    for (int i = 0; i < 9; i++) {
        if (i < 3) ref_t_inv_float[i] = static_cast<float>(ref.p.t_inv[i]);
        ref_K_inv_float[i] = static_cast<float>(ref.p.K_inv[i]);
        ref_R_inv_float[i] = static_cast<float>(ref.p.R_inv[i]);
    }

    // Allocate and copy reference image
    unsigned char* d_ref_Y;
    int ref_stride = ref.YUV[0].step[0];
    cudaMalloc(&d_ref_Y, height * ref_stride * sizeof(unsigned char));
    cudaMemcpy(d_ref_Y, ref.YUV[0].data, height * ref_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

    for (auto& cam : cam_vector) {
        if (cam.name == ref.name) {
            continue;  // Skip reference camera
        }
        std::cout << "Processing camera: " << cam.name << std::endl;

        // Convert camera matrices to float
        std::vector<float> cam_K_float(9), cam_R_float(9), cam_t_float(3);
        for (int i = 0; i < 9; i++) {
            if (i < 3) cam_t_float[i] = static_cast<float>(cam.p.t[i]);
            cam_K_float[i] = static_cast<float>(cam.p.K[i]);
            cam_R_float[i] = static_cast<float>(cam.p.R[i]);
        }

        // Matrix multiplication cam_R * ref_R_inv * ref_K_inv
        std::vector<float> temporary(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += cam_R_float[i * 3 + k] * ref_R_inv_float[k * 3 + j];
                }
                temporary[i * 3 + j] = sum;
            }
        }

        std::vector<float> R_cam_RK_ref(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float sum = 0.0;
                for (int k = 0; k < 3; k++) {
                    sum += temporary[i * 3 + k] * ref_K_inv_float[k * 3 + j];
                }
                R_cam_RK_ref[i * 3 + j] = sum;
            }
        }

        std::vector<float> RT_cam_T_ref(3);
        for (int i = 0; i < 3; i++) {
            float sum = 0.0;
            for (int j = 0; j < 3; j++) {
                sum += cam_R_float[i * 3 + j] * cam_t_float[j];
            }
            RT_cam_T_ref[i] = sum + ref_t_inv_float[i];
        }

        // Allocate and copy matrices to constant memory
        cudaMemcpyToSymbol(d_cam_K, cam_K_float.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_R_cam_RK_ref, R_cam_RK_ref.data(), 9 * sizeof(float));
        cudaMemcpyToSymbol(d_RT_cam_T_ref, RT_cam_T_ref.data(), 3 * sizeof(float));

        // Allocate and copy camera image
        unsigned char* d_cam_Y;
        int cam_stride = cam.YUV[0].step[0];
        cudaMalloc(&d_cam_Y, cam.height * cam_stride * sizeof(unsigned char));
        cudaMemcpy(d_cam_Y, cam.YUV[0].data, cam.height * cam_stride * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Set 2D block dimensions
        dim3 blockDim(32, 32);
        //dim3 blockDim(16, 16);

        // Calculate padding for shared memory
        int padding = window / 2;

        // Calculate shared memory size for reference image with padding
        int smem_width = blockDim.x + 2 * padding;
        int smem_height = blockDim.y + 2 * padding;
        size_t smem_size = smem_width * smem_height * sizeof(unsigned char);

        // Set 3D grid dimensions
        dim3 gridDim(
            (width + blockDim.x - 1) / blockDim.x,
            (height + blockDim.y - 1) / blockDim.y,
            ZPlanes  // One grid layer per Z plane
        );

        // Launch kernel with 3D grid, 2D blocks, and shared memory
        auto start = std::chrono::high_resolution_clock::now();

        grid3d_shared_ref_2_kernel << <gridDim, blockDim, smem_size >> > (
            width, height, d_ref_Y,
            cam.width, cam.height, d_cam_Y,
            d_cost_cube, window,
            static_cast<float>(ZNear), static_cast<float>(ZFar), ZPlanes
            );

        auto end = std::chrono::high_resolution_clock::now();
        // Print time duration of algorithm
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() * 1000 << " ms" << std::endl;

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }

        // Synchronize to ensure kernel execution completes
        cudaDeviceSynchronize();

        // Free camera resources
        cudaFree(d_cam_Y);
    }

    // Copy results back to host
    cudaMemcpy(cost_cube_data.data(), d_cost_cube, cost_cube_size, cudaMemcpyDeviceToHost);

    // Convert to OpenCV matrices
    std::vector<cv::Mat> result(ZPlanes);
    for (int i = 0; i < ZPlanes; ++i) {
        result[i] = cv::Mat(height, width, CV_32FC1, cost_cube_data.data() + i * width * height).clone();
    }

    // Free reference resources
    cudaFree(d_ref_Y);
    cudaFree(d_cost_cube);

    return result;
}