//
// Created by leo on 27.02.2021.
//

#include "SiftGPU.h"
#include "vector"
#include "cassert"
#include "iostream"
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    const char *pathArg;
    if (argc != 2) {
        std::cout << "Provide path to image as only argument" << std::endl;
        return -1;
    } else {
        pathArg = argv[1];
    }
    SiftGPU sift_gpu;
//    std::vector<char *> siftGpuArgsStrings = {"-fo", "-1", "-v", "1"}; // works fine
    std::vector<char *> siftGpuArgsStrings = {"-cuda", "0", "-fo", "-1", "-v", "1"}; // give cuda error ?
//    std::vector<char *> siftGpuArgs;

//    for (int i = 0; i < siftGpuArgsStrings.size(); ++i) {
//        siftGpuArgs.push_back(siftGpuArgsStrings[i].data());
//    }
    char **data = siftGpuArgsStrings.data();
//    sift_gpu.CreateContextGL();
    sift_gpu.ParseParam(siftGpuArgsStrings.size(), siftGpuArgsStrings.data());
    assert(sift_gpu.VerifyContextGL() == sift_gpu.SIFTGPU_FULL_SUPPORTED);
    std::string path = "/home/leo/Desktop/1305031453.059754.png";
    const char *pathData = path.data();
    sift_gpu.RunSIFT(pathArg);


//  SiftMatchGPU sift_match_gpu;

    std::cout << "hello!" << std::endl;


    int num1 = sift_gpu.GetFeatureNum();
    std::vector<float> descriptors1(128 * num1);
    std::vector<SiftGPU::SiftKeypoint> keys1(num1);
    sift_gpu.GetFeatureVector(keys1.data(), descriptors1.data());


    std::string pathToRGBImage = path;

    cv::Mat imageNoKeyPoint = cv::imread(pathToRGBImage, cv::IMREAD_COLOR);
    std::vector<cv::KeyPoint> keyPointsToShow;


    cv::Mat imageWithKeyPoint = cv::imread(pathToRGBImage, cv::IMREAD_COLOR);

    for (const auto &keyPointInfo: keys1) {
        double x = keyPointInfo.x;
        double y = keyPointInfo.y;
        cv::KeyPoint keyPointToShow(x, y, keyPointInfo.s);

        keyPointsToShow.push_back(keyPointToShow);
    }

    cv::drawKeypoints(imageNoKeyPoint, {keyPointsToShow}, imageWithKeyPoint);

    std::string pathToSaveWithName = "keypoints.png";
    cv::imwrite(pathToSaveWithName, imageWithKeyPoint);
    std::cout << "done" << std::endl;

    return 0;
}