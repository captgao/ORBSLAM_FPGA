#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#define EXTR_REG_BASE 0x80000000
#define ORB_BASE 0x82000000
#define DESC_BASE 0x84000000
#define WR_PADDR 0x38000000
#define RD_PADDR 0x38800000UL
using namespace cv;
using namespace std;

namespace ORB_SLAM2{

struct FPGA_ORBResult{
    int batch;
    unsigned char aa;
    unsigned char sumrow[3];
    unsigned char sumcol[3];
    unsigned int pixid;
    unsigned char fast;

};
struct FPGA_Descriptor{
    unsigned char d[32];
};

void FPGAExtract(unsigned char* mat, unsigned int height, unsigned int width, unsigned int batch){
    printf("FPGAExtract start \n");
    int dev_fd;
	dev_fd = open("/dev/mem",O_RDWR|O_SYNC);
    unsigned char *toWrite = (unsigned char*) mmap(NULL,409600,PROT_READ|PROT_WRITE,
	MAP_SHARED,dev_fd,WR_PADDR);
    unsigned char *toRead = (unsigned char*) mmap(NULL,409600,PROT_READ|PROT_WRITE,
	MAP_SHARED,dev_fd,RD_PADDR);
    void* map_base = (unsigned char*) mmap(NULL,4096UL,PROT_READ|PROT_WRITE,
	MAP_SHARED,dev_fd,EXTR_REG_BASE);
    unsigned int *toReadaddr,*readlen,*toWriteaddr,*imgsize,*reset,*finish,*ctrl,*batchreg;
	unsigned int *debug, *deviceid;
    toWriteaddr = (unsigned int*)map_base;
	readlen = (unsigned int*)(map_base + 1*4);
	toReadaddr = (unsigned int*)(map_base + 2*4);
	imgsize = (unsigned int*)(map_base + 3*4);
    batchreg = (unsigned int*)(map_base + 4*4);
	debug = (unsigned int*) (map_base + 11*4);
	deviceid = (unsigned int*) (map_base + 10*4);
	reset = (unsigned int*)(map_base + 13*4);
	finish = (unsigned int*)(map_base + 14*4);
	ctrl = (unsigned int*)(map_base + 15*4);
//    printf("device identity %d %x\n",*deviceid, *debug);
    *reset=1;
    usleep(1000);
    *reset=0;

    *toWriteaddr = WR_PADDR;
    *toReadaddr = RD_PADDR;
    *readlen = (width << 16) + width * 2;
    *imgsize = (height << 16) + width;
//    *readlen = 0x00280050;
//    *imgsize = 0x00400028;
    *batchreg = batch;
    *toReadaddr = RD_PADDR;
    printf("readlen %x, imgsize %x\n",*readlen,*imgsize);
 //   printf("0x%x 0x%x %x",*toWriteaddr,*toReadaddr,RD_PADDR);

    memcpy(toWrite,mat,height*width);
    syscall(399,(void*)toWrite,409600);
    syscall(400,WR_PADDR,409600);
    syscall(401,WR_PADDR,409600);
    *ctrl = 1;
    usleep(1);
    *ctrl = 0;
 //   printf("ctrl %d\n\n",*ctrl);
    while(*finish == 0){
//	printf("%x\n",*debug);
    };
    if(dev_fd)
		close(dev_fd);
	munmap(map_base,4096UL);
    munmap(toWrite, 409600);
    munmap(toRead, 409600);
    printf("FPGAExtract end\n");
}

void FPGAResult(vector<KeyPoint>& _keypoints, OutputArray& _descriptors){
    printf("FPGAResult start\n");
    int dev_fd;
	dev_fd = open("/dev/mem",O_RDWR|O_SYNC);
    struct FPGA_ORBResult * orbstart = (struct FPGA_ORBResult*) mmap(NULL,1048576,PROT_READ,MAP_SHARED,dev_fd,ORB_BASE);
    struct FPGA_Descriptor * descstart = (struct FPGA_Descriptor*) mmap(NULL,1048576,PROT_READ,MAP_SHARED,dev_fd,DESC_BASE);
    for(int i =0; i < 1024; i++,orbstart++){
        if(orbstart->fast != 0){
            int* tmp;
            tmp =(int*) orbstart -> sumrow;
            int sumrow = *tmp;
            if(sumrow & 0x00800000){
                sumrow = sumrow | 0xff000000;
            }
            else{
                sumrow = sumrow & 0x00ffffff;
            }
            tmp = (int*)orbstart -> sumcol;
            int sumcol = *tmp;
            if(sumcol & 0x00800000){
                sumcol = sumcol | 0xff000000;
            }
            else{
                sumcol = sumcol & 0x00ffffff;
            }
            float angle = fastAtan2((float)sumrow,(float)sumcol);
            _keypoints.push_back(KeyPoint((float)orbstart->pixid,0.f,7.f,angle,orbstart->fast));
        }
    }
    printf("FPGAResult end\n");
}

void FPGACvtMat(cv::Mat cvmat, unsigned char* mat){

    for(int i = 0; i < cvmat.cols; i+=8){
        for(int j = 0;j < cvmat.rows; j++){
            for(int k=0;k<8;k++){
                *mat = cvmat.at<uchar>(j,k+i);
                mat++;
            }
        }
    }
}
}
