#include <iostream>
#include <opencv2/opencv.hpp>

int main(int, char**)
{
    std::cout << "Starting OpenCV\n";
    auto filename = "c:/Users/Admin/Downloads/lenna (1).png";
    auto image  = cv::imread(filename);

    cv::imshow("Image",image);
    cv::waitKey();
    
}
