

#include <cuda.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <device_functions.h>
#include "costum_library/costumlib.h"

__constant__ char Filter_D[25];

__global__ void Blur(uchar* rgb, uchar* sortie, std::size_t cols, std::size_t rows, uchar paddingSize) {
    //essayer d'enlever le holder
    // Set the padding size and filter size

    unsigned int filterSize = (2 * paddingSize) + 1;
    int paddedW = 2 * paddingSize + cols;
    int paddedH = 2 * paddingSize + rows;
    auto  j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
    auto  i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;


    // The multiply-add operation for the pixel coordinate ( j, i )
    if (j >= paddingSize && j < paddedW - paddingSize && i >= paddingSize && i < paddedH - paddingSize) {
        auto oPixelPos = (i - paddingSize) * cols + (j - paddingSize);


        sortie[oPixelPos] = 0.0;

        auto channel = 0.0;

        auto iPixelPos = 0;
        auto coefPos = 0;
        for (int k = -paddingSize; k <= paddingSize; k++) {
            for (int l = -paddingSize; l <= paddingSize; l++) {

                iPixelPos = (i + k) * paddedW + (j + l);
                coefPos = (k + paddingSize) * filterSize + (l + paddingSize);
                channel += (rgb[iPixelPos] * Filter_D[coefPos]) / 25;


            }
            sortie[oPixelPos] = channel;
        }


        //sortie[oPixelPos] += sqrt((float)vvhh);

    }
}

int main()
{
    cv::Mat m_in = cv::imread("images/Lenna.jpg", cv::IMREAD_GRAYSCALE);
    //creation d'une mat qui contiendra les channels apres traitement 
    //


    auto S = 2;


    auto rows = m_in.rows;
    auto cols = m_in.cols;
    auto paddedW = 2 * S + cols;
    auto paddedH = 2 * S + rows;
    std::vector<unsigned char > g(cols * rows);
    std::vector<unsigned char > bor(paddedW * paddedH);

    //faire en sorte de detecter les types d'images 
    cv::Mat m_out(rows, cols, CV_8UC1, g.data());
    cv::Mat m_bor(rows, cols, CV_8UC1, bor.data());


    cv::copyMakeBorder(m_in, m_bor, S, S, S, S, CV_HAL_BORDER_CONSTANT, 0);
    auto rows_bo = m_bor.rows;
    auto cols_bo = m_bor.cols;

    auto rgb_bo = m_bor.data;


    uchar* sob_h;


    //result from device
    uchar* sob_d;


    /* { 1, 4, 7, 4, 1,
                         4,16, 26, 16, 4,
                         7,26,41, 26,7,
                         4,16,26,16,4,
                         1, 4, 7, 4, 1 };*/
    char Filter_h[25] = { 1, 1, 1, 1, 1,1,1,
                         1,1,1, 1, 1, 1, 1,1,1,
                         1,1, 1, 1, 1, 1,1,1,
                         1 };

    HANDLE_ERROR(cudaMemcpyToSymbol(Filter_D, Filter_h, 25));

    dim3 t(32, 32);
    dim3 b((cols_bo - 1) / t.x + 1, (rows_bo - 1) / t.y + 1);
    cudaEvent_t start, stop;
    int count = 0;
    float elapsedTime;

    ////////////////////////
  //  while (count <= 1000) {
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
        HANDLE_ERROR(cudaMalloc(&sob_h, rows_bo * cols_bo));
        HANDLE_ERROR(cudaMalloc(&sob_d, rows_bo * cols_bo));
        // HANDLE_ERROR(cudaHostAlloc((void**)&sob_d, rows * cols * sizeof(*sob_d), cudaHostAllocDefault));

        HANDLE_ERROR(cudaMemcpy(sob_h, rgb_bo, rows_bo * cols_bo, cudaMemcpyHostToDevice));
        Blur << < b, t >> > (sob_h, sob_d, cols, rows, S);
        HANDLE_ERROR(cudaMemcpy(g.data(), sob_d, rows * cols, cudaMemcpyDeviceToHost));
        // HANDLE_ERROR(cudaFreeHost(sob_d));
        HANDLE_ERROR(cudaFree(sob_h));
        HANDLE_ERROR(cudaFree(sob_d));

        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));

        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
            start, stop));
        printf("%3.1f\n", elapsedTime);
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
        count++;
   // }
    //merging the channels
    cv::imshow("f", m_out);
    cv::waitKey();
    cv::imwrite("C:/Users/lagab/Pictures/Lenna_blur_5per5.jpg", m_out);


    return 0;
}