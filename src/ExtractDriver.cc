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

namespace ORB_SLAM2
{

struct FPGA_ORBResult
{
    int batch;
    unsigned int sumrow24;
    unsigned char sumcol[3];
    unsigned char fast;
    unsigned int pixid;
};
struct FPGA_Descriptor
{
    unsigned char d[32];
};

void FPGAExtract(unsigned char *mat, unsigned int height, unsigned int width, unsigned int batch)
{
    //    printf("FPGAExtract start \n");
    int dev_fd;
    dev_fd = open("/dev/mem", O_RDWR | O_SYNC);
    unsigned char *toWrite = (unsigned char *)mmap(NULL, 409600, PROT_READ | PROT_WRITE,
                                                   MAP_SHARED, dev_fd, WR_PADDR);
    unsigned char *toRead = (unsigned char *)mmap(NULL, 409600, PROT_READ | PROT_WRITE,
                                                  MAP_SHARED, dev_fd, RD_PADDR);
    unsigned char *map_base = (unsigned char *)mmap(NULL, 4096UL, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, dev_fd, EXTR_REG_BASE);
    unsigned int *toReadaddr, *readlen, *toWriteaddr, *imgsize, *reset, *finish, *ctrl, *batchreg;
    toWriteaddr = (unsigned int *)map_base;
    readlen = (unsigned int *)(map_base + 1 * 4);
    toReadaddr = (unsigned int *)(map_base + 2 * 4);
    imgsize = (unsigned int *)(map_base + 3 * 4);
    batchreg = (unsigned int *)(map_base + 4 * 4);

    reset = (unsigned int *)(map_base + 13 * 4);
    finish = (unsigned int *)(map_base + 14 * 4);
    ctrl = (unsigned int *)(map_base + 15 * 4);
    *reset = 1;
    usleep(1);
    *reset = 0;

    *toWriteaddr = WR_PADDR;
    *toReadaddr = RD_PADDR;
    *readlen = (width << 16) + width * 2;
    *imgsize = (height << 16) + width;
    *batchreg = batch;
    *toReadaddr = RD_PADDR;
    printf("readlen %x, imgsize %x\n", *readlen, *imgsize);

    memcpy(toWrite, mat, height * width);
    syscall(399, (void *)toWrite, 409600);
    syscall(400, WR_PADDR, 409600);
    syscall(401, WR_PADDR, 409600);
    *ctrl = 1;
    while (*ctrl != 0)
    {
        usleep(1);
    };
    if (dev_fd)
        close(dev_fd);
    munmap(map_base, 4096UL);
    munmap(toWrite, 409600);
    munmap(toRead, 409600);
    //    printf("FPGAExtract end\n");
}

void FPGAResult(vector<KeyPoint> &_keypoints, OutputArray &_descriptors, int level)
{
    //    printf("FPGAResult start\n");
    static FILE *fp;
    printf("size %d\n", sizeof(struct FPGA_ORBResult));
    int dev_fd;
    dev_fd = open("/dev/mem", O_RDWR | O_SYNC);
    unsigned char *map_base = (unsigned char *)mmap(NULL, 4096UL, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, dev_fd, EXTR_REG_BASE);
    int *heap_outstart = (int *)(map_base + 20);
    *heap_outstart = 1;
    while (*heap_outstart != 0)
    {
        	usleep(1);
    }
    struct FPGA_ORBResult *orb = (struct FPGA_ORBResult *)mmap(NULL, 1048576, PROT_READ, MAP_SHARED, dev_fd, ORB_BASE);
    struct FPGA_Descriptor *descstart = (struct FPGA_Descriptor *)mmap(NULL, 1048576, PROT_READ, MAP_SHARED, dev_fd, DESC_BASE);
    for (int i = 0; i < 1024; i++)
    {
        struct FPGA_ORBResult *orbstart = orb + i;
        if (orbstart->fast != 0)
        {
            printf("get! %d\n", orbstart->batch);
            int sumrow = orbstart->sumrow24;
            if (sumrow & 0x00800000)
            {
                sumrow = sumrow | 0xff000000;
            }
            else
            {
                sumrow = sumrow & 0x00ffffff;
            }
            int *tmp = (int *)orbstart->sumcol;
            int sumcol = *tmp;
            if (sumcol & 0x00800000)
            {
                sumcol = sumcol | 0xff000000;
            }
            else
            {
                sumcol = sumcol & 0x00ffffff;
            }
            float angle = fastAtan2((float)sumrow, (float)sumcol);
            _keypoints.push_back(KeyPoint((float)orbstart->pixid, (float)level, (float)orbstart->batch, angle, orbstart->fast));
            if (fp == NULL)
                fp = fopen("/root/desc.txt", "w");
            struct FPGA_Descriptor *curd = descstart + i;
            for (int i = 0; i < 32; i++)
            {
                fprintf(fp, "%2x ", curd->d[i]);
            }
            fprintf(fp, "\n");
        }
    }
//    printf("FPGAResult end\n");
    munmap(orb, 1048576);
    munmap(descstart, 1048576);
}

void FPGACvtMat(cv::Mat cvmat, unsigned char *mat)
{
    unsigned char *orgmat = mat;
    for (int i = 0; i < cvmat.rows; i += 8)
    {
        for (int j = 0; j < cvmat.cols; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                *mat = cvmat.at<uchar>(k + i, j);
                mat++;
            }
        }
    }

}
} // namespace ORB_SLAM2
