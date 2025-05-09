#include "main.cuh"
#include <cstdio>
#include <cmath>

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
