
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>


#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

int main(int argc, char **argv)
{
    ofstream out("/root/out.txt");
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    ORB_SLAM2::FPGAExtract(0,0,0,0);
    ORB_SLAM2::FPGAResult(mvKeys,mDescriptors,0);
    for(auto it : mvKeys){
        out << it.pt.x << "," << it.pt.y << endl;
        out << it.angle << " " << it.response << " " << it.size << " " << it.class_id <<endl;
        out << endl;
    }
    out.close();
    cout << "successfully finished" << endl;
}
