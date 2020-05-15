

#include <cuda.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <device_functions.h>
#include "costum_library/costumlib.h"

__constant__ char Filter[9];

__global__ void BlurOneKernel(uchar* rgb_b, uchar* rgb_g, uchar* rgb_r, uchar* sortie_b, uchar* sortie_g, uchar* sortie_r, std::size_t cols, std::size_t rows, uchar paddingSize) {
    //essayer d'enlever le holder
    // Set the padding size and filter size
    //int S = Ss;
    //int paddingSize = S;
    unsigned int filterSize = (2 * paddingSize) + 1;
    int paddedW = 2 * paddingSize + cols;
    int paddedH = 2 * paddingSize + rows;
   


    //char Filter[9]={1, 1, 1, 1, 1, 1, 1, 1, 1};


    // Set the pixel coordinate

    //bizzare , lorsque je fais j=i+paddingsize (รงa m'affiche noir)
    auto  j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
    auto  i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

    // The multiply-add operation for the pixel coordinate ( j, i )
    if (j >= paddingSize && j < paddedW - paddingSize && i >= paddingSize && i < paddedH - paddingSize) {
        auto oPixelPos = (i - paddingSize) * cols + (j - paddingSize);
        sortie_b[oPixelPos] = 0.0;
        sortie_g[oPixelPos] = 0.0;
        sortie_r[oPixelPos] = 0.0;
        auto channelB = 0.0;
        auto  channelG = 0.0;
        auto  channelR = 0.0;

        for (int k = -paddingSize; k <= paddingSize; k++) {

            for (int l = -paddingSize; l <= paddingSize; l++) {
                //le probleme est la 
                auto iPixelPos = (i + k) * paddedW + (j + l);

                auto coefPos = (k + paddingSize) * filterSize + (l + paddingSize);
                channelB += (rgb_b[iPixelPos] * Filter[coefPos]) / 9;
                channelG += (rgb_g[iPixelPos] * Filter[coefPos]) / 9;
                channelR += (rgb_r[iPixelPos] * Filter[coefPos]) / 9;

            }
            //enlever ça et remplacer plus haut
            sortie_b[oPixelPos] = channelB;
            sortie_g[oPixelPos] = channelG;
            sortie_r[oPixelPos] = channelR;

        }

    }

}


int main()
{
    cv::Mat m_in = cv::imread("images/Lenna.jpg", cv::IMREAD_COLOR);
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
    int count = 0;
      
    while (count <= 1000) {
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));

        HANDLE_ERROR(cudaMalloc(&blue_h, rows_bo * cols_bo));
        HANDLE_ERROR(cudaMalloc(&green_h, rows_bo * cols_bo));
        HANDLE_ERROR(cudaMalloc(&red_h, rows_bo * cols_bo));

        HANDLE_ERROR(cudaHostAlloc((void**)&blue_d, rows * cols * sizeof(*blue_d), cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc((void**)&green_d, rows * cols * sizeof(*green_d), cudaHostAllocDefault));
        HANDLE_ERROR(cudaHostAlloc((void**)&red_d, rows * cols * sizeof(*red_d), cudaHostAllocDefault));

      
        HANDLE_ERROR(cudaMemcpy(blue_h, bl, rows_bo * cols_bo, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(green_h, gr, rows_bo * cols_bo, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(red_h, re, rows_bo * cols_bo, cudaMemcpyHostToDevice));
        

        BlurOneKernel << < b, t >> > (blue_h, green_h, red_h, blue_d, green_d, red_d, cols, rows, S);
        //cudaDeviceSynchronize();

        HANDLE_ERROR(cudaMemcpy(blue.data(), blue_d, rows * cols, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(green.data(), green_d, rows * cols, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(red.data(), red_d, rows * cols, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaFree(blue_h));
        HANDLE_ERROR(cudaFree(green_h));
        HANDLE_ERROR(cudaFree(red_h));
        mergechannels[0].data = blue.data();
        mergechannels[1].data = green.data();
        mergechannels[2].data = red.data();
        HANDLE_ERROR(cudaFreeHost(blue_d));
        HANDLE_ERROR(cudaFreeHost(green_d));
        HANDLE_ERROR(cudaFreeHost(red_d));

        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
            start, stop));
        printf("%3.1f\n", elapsedTime);
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
        count++;
    }
    //merging the channels
    cv::merge(mergechannels, 3, m_out);
    cv::imshow("f", m_out);
    cv::waitKey();
    cv::imwrite("C:/Users/lagab/Pictures/2ou4t23.jpg", m_out);



    return 0;
}
