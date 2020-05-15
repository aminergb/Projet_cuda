

#include <cuda.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <device_functions.h>
#include "costum_library/costumlib.h"

int main()
{




    int count = 0;
  //  while (count <= 1000) {
        auto startcpu = std::chrono::steady_clock::now();
        cv::Mat m_in = cv::imread("images//Lenna.jpg", cv::IMREAD_COLOR);
        //creation d'une mat qui contiendra les channels apres traitement 
        //
        cv::Mat mergechannels[3];
        cv::split(m_in, mergechannels);
        auto blm = mergechannels[0].data;
        auto grm = mergechannels[1].data;
        auto rem = mergechannels[2].data;

        auto paddingSize= 1;
        auto S = 1;
        auto rgb = m_in.data;
        auto rows = m_in.rows;
        auto cols = m_in.cols;
        auto paddedW = 2 * paddingSize + cols;
        auto paddedH = 2 * paddingSize+ rows;
        std::vector<unsigned char > g(cols * rows);
        std::vector<unsigned char > bor(paddedW * paddedH);
        
        //faire en sorte de detecter les types d'images 
        cv::Mat m_out(rows, cols, CV_8UC1, g.data());
        cv::Mat m_bor(rows, cols, CV_8UC1, bor.data());
      

        cv::copyMakeBorder(m_in, m_bor, S, S, S, S, CV_HAL_BORDER_CONSTANT, 0);
        auto rows_bo = m_bor.rows;
        auto cols_bo = m_bor.cols;

        auto rgb_bo = m_bor.data;
        cv::Mat splitchannels[3];

        cv::split(m_bor, splitchannels);
        auto bl = splitchannels[0].data;
        auto gr = splitchannels[1].data;
        auto re = splitchannels[2].data;

     


        uchar mask_h[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        auto filterSize = (2 * paddingSize) + 1;
        

        for (unsigned int i = paddingSize; i < paddedH - paddingSize; i++) {
            for (unsigned int j = paddingSize; j < paddedW - paddingSize; j++) {

                // The multiply-add operation for the pixel coordinate ( j, i )
                unsigned int oPixelPos = (i - paddingSize) * cols + (j - paddingSize);
               blm[oPixelPos] = 0.0;
               grm[oPixelPos] = 0.0;
               rem[oPixelPos] = 0.0;
                for (int k = -S; k <= S; k++) {
                    for (int l = -S; l <= S; l++) {
                        unsigned int iPixelPos = (i + k) * paddedW + (j + l);
                        unsigned int coefPos = (k + S) * filterSize + (l + S);
                        blm[oPixelPos] += bl[iPixelPos] *mask_h[coefPos]/9;
                       grm[ oPixelPos] += gr[ iPixelPos] * mask_h[coefPos]/9;
                        rem[oPixelPos] += re[ iPixelPos] * mask_h[coefPos]/9;
                    }
                }

            }
        }

        mergechannels[0].data = blm;
        mergechannels[1].data = grm;
        mergechannels[2].data = rem;
        float elapsedTime;
       

        //merging the channels
        cv::merge(mergechannels, 3, m_out);
        cv::imwrite("C:/Users/lagab/Pictures/2ou4t23.jpg", m_out);
        auto endcpu = std::chrono::steady_clock::now();
        float elapsedTimeCPU = std::chrono::duration_cast<std::chrono::milliseconds>(endcpu - startcpu).count();
        printf("%3.1f\n", elapsedTimeCPU);

        count++;

  //  }
     cv::imshow("f", m_out);
    cv::waitKey();




    return 0;
}
