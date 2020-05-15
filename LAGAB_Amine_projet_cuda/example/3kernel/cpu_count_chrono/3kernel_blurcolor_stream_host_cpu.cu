

#include <cuda.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <device_functions.h>
#include "costum_library/costumlib.h"

__constant__ char Filter[9];

__global__ void BlurColorKernel(uchar* rgb, uchar* sortie, std::size_t cols, std::size_t rows, uchar paddingSize) {
    //essayer d'enlever le holder
    // Set the padding size and filter size

    unsigned int filterSize = (2 * paddingSize) + 1;
    int paddedW = 2 * paddingSize + cols;
    int paddedH = 2 * paddingSize + rows;
    //filter testing
    //{1, 1, 1, 1, 1, 1, 1, 1, 1}


    // Set the pixel coordinate

    //bizzare , lorsque je fais j=i+paddingsize (รงa m'affiche noir)
    auto  j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
    auto  i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

    // The multiply-add operation for the pixel coordinate ( j, i )
    if (j >= paddingSize && j < paddedW - paddingSize && i >= paddingSize && i < paddedH - paddingSize) {
        auto oPixelPos = (i - paddingSize) * cols + (j - paddingSize);
        sortie[oPixelPos] = 0.0;
        auto channel = 0.0;
        for (int k = -paddingSize; k <= paddingSize; k++) {
            for (int l = -paddingSize; l <= paddingSize; l++) {
                //le probleme est la 
                auto iPixelPos = (i + k) * paddedW + (j + l);
                auto coefPos = (k + paddingSize) * filterSize + (l + paddingSize);
                channel += (rgb[iPixelPos] * Filter[coefPos]) / 9;
            }
            sortie[oPixelPos] = channel;
        }

    }
}


int main()
{
    int count = 0;
    while (count <= 1000) {

        auto startcpu = std::chrono::steady_clock::now();
    cv::Mat m_in = cv::imread("C:/Users/lagab/Pictures/Lenna.jpg", cv::IMREAD_COLOR);
    //creation d'une mat qui contiendra les channels apres traitement 
    //
    cv::Mat mergechannels[3];
    cv::split(m_in, mergechannels);

    auto S = 1;

    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    auto paddedW = 2 * S + cols;
    auto paddedH = 2 * S + rows;
    std::vector<unsigned char > g(cols * rows);
    std::vector<unsigned char > bor(paddedW * paddedH);
    std::vector<unsigned char > blue(cols * rows);
    std::vector<unsigned char > green(cols * rows);
    std::vector<unsigned char > red(cols * rows);

    //faire en sorte de detecter les types d'images 
    cv::Mat m_out(rows, cols, CV_8UC1, g.data());
    cv::Mat m_bor(rows, cols, CV_8UC1, bor.data());
    cv::Mat m_blue(rows, cols, CV_8UC1, blue.data());
    cv::Mat m_green(rows, cols, CV_8UC1, green.data());
    cv::Mat m_red(rows, cols, CV_8UC1, red.data());


    cv::copyMakeBorder(m_in, m_bor, S, S, S, S, CV_HAL_BORDER_CONSTANT, 0);
    auto rows_bo = m_bor.rows;
    auto cols_bo = m_bor.cols;

    auto rgb_bo = m_bor.data;
    cv::Mat splitchannels[3];

    cv::split(m_bor, splitchannels);
    auto bl = splitchannels[0].data;
    auto gr = splitchannels[1].data;
    auto re = splitchannels[2].data;

    uchar* blue_h;
    uchar* green_h;
    uchar* red_h;

    //result from device
    uchar* blue_d;
    uchar* green_d;
    uchar* red_d;



    char mask_h[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    HANDLE_ERROR(cudaMemcpyToSymbol(Filter, mask_h, 9));


    //allocation vers gpu
    dim3 t(32, 32);
    dim3 b((cols_bo - 1) / t.x + 1, (rows_bo - 1) / t.y + 1);
    cudaEvent_t start, stop;
    cudaStream_t streamblue, streamgreen, streamred;
   

        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
        HANDLE_ERROR(cudaStreamCreate(&streamblue));
        HANDLE_ERROR(cudaStreamCreate(&streamgreen));
        HANDLE_ERROR(cudaStreamCreate(&streamred));
        HANDLE_ERROR(cudaMalloc(&blue_h, rows_bo * cols_bo));
        HANDLE_ERROR(cudaMalloc(&green_h, rows_bo * cols_bo));
        HANDLE_ERROR(cudaMalloc(&red_h, rows_bo * cols_bo));

        HANDLE_ERROR(cudaHostAlloc((void**)&blue_d, rows * cols * sizeof(*blue_d), cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc((void**)&green_d, rows * cols * sizeof(*green_d), cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc((void**)&red_d, rows * cols * sizeof(*red_d), cudaHostAllocDefault));


        HANDLE_ERROR(cudaMemcpyAsync(blue_h, bl, rows_bo * cols_bo, cudaMemcpyHostToDevice, streamblue));
        HANDLE_ERROR(cudaMemcpyAsync(green_h, gr, rows_bo * cols_bo, cudaMemcpyHostToDevice, streamgreen));
        HANDLE_ERROR(cudaMemcpyAsync(red_h, re, rows_bo * cols_bo, cudaMemcpyHostToDevice, streamred));

        BlurColorKernel << < b, t, 0, streamblue >> > (blue_h, blue_d, cols, rows, S);
        BlurColorKernel << < b, t, 0, streamgreen >> > (green_h, green_d, cols, rows, S);
        BlurColorKernel << < b, t, 0, streamred >> > (red_h, red_d, cols, rows, S);

        HANDLE_ERROR(cudaMemcpyAsync(blue.data(), blue_d, rows * cols, cudaMemcpyDeviceToHost, streamblue));
        HANDLE_ERROR(cudaMemcpyAsync(green.data(), green_d, rows * cols, cudaMemcpyDeviceToHost, streamgreen));
        HANDLE_ERROR(cudaMemcpyAsync(red.data(), red_d, rows * cols, cudaMemcpyDeviceToHost, streamred));
        HANDLE_ERROR(cudaStreamSynchronize(streamblue));
        HANDLE_ERROR(cudaStreamSynchronize(streamgreen));
        HANDLE_ERROR(cudaStreamSynchronize(streamred));
        HANDLE_ERROR(cudaFree(blue_h));
        HANDLE_ERROR(cudaFree(green_h));
        HANDLE_ERROR(cudaFree(red_h));
        mergechannels[0].data = blue.data();
        mergechannels[1].data = green.data();
        mergechannels[2].data = red.data();
        HANDLE_ERROR(cudaFreeHost(blue_d));
        HANDLE_ERROR(cudaFreeHost(green_d));
        HANDLE_ERROR(cudaFreeHost(red_d));
        HANDLE_ERROR(cudaStreamDestroy(streamblue));
        HANDLE_ERROR(cudaStreamDestroy(streamgreen));
        HANDLE_ERROR(cudaStreamDestroy(streamred));
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
            start, stop));
      //  printf("%3.1f\n", elapsedTime);
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
       
    //merging the channels
    cv::merge(mergechannels, 3, m_out);
    cv::imwrite("C:/Users/lagab/Pictures/2ou4t23.jpg", m_out);
    auto endcpu = std::chrono::steady_clock::now();
    float elapsedTimeCPU = std::chrono::duration_cast<std::chrono::milliseconds>(endcpu - startcpu).count();
    printf("%3.1f\n", elapsedTimeCPU);

    count++;
    }

   // cv::imshow("f", m_out);
    cv::waitKey();
   


    return 0;
}
