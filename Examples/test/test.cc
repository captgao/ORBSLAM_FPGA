
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>


#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

int main(int argc, char **argv)
{
    cv::Mat im = cv::imread("/root/test.png",cv::IMREAD_GRAYSCALE);
    ofstream out("/root/out.txt");
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    ORB_SLAM2::ORBextractor* extractor = new ORB_SLAM2::ORBextractor(1000,1.2,8,20,7);
    (*extractor)(im,cv::Mat(),mvKeys,mDescriptors);
    
    for(auto it : mvKeys){
        out << it.pt.x << "," << it.pt.y << endl;
        out << it.angle << " " << it.response << " " << it.octave << " " << it.class_id <<endl;
        out << endl;
    }
    out.close();
    cout << "successfully finished" << endl;
}
