#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <opencv2/opencv.hpp>
#define EXTR_REG_BASE (unsigned int*)0x43C00000
#define MATCH_REG_BASE (int*)0X43C10000

int main(){
	int dev_fd;
	dev_fd = open("/dev/mem",O_RDWR|O_SYNC);
	unsigned int WR_PADDR = 0x38000000;
	unsigned int RD_PADDR = 0X39000000;
	unsigned char *wrmem = (unsigned char*) mmap(NULL,40960,PROT_READ|PROT_WRITE,
	MAP_SHARED,dev_fd,WR_PADDR);
	unsigned char *rdmem = (unsigned char*) mmap(NULL,40960,PROT_READ|PROT_WRITE,
	MAP_SHARED,dev_fd,RD_PADDR);
	unsigned int *readaddr,*readlen,*writeaddr,*imgsize,*reset,*finish,*ctrl;
	readaddr = EXTR_REG_BASE;
	readlen = (unsigned int*)(EXTR_REG_BASE  + 1*4);
	writeaddr = (unsigned int*)(EXTR_REG_BASE + 2*4);
	imgsize = (unsigned int*)(EXTR_REG_BASE + 3*4);
	reset = (unsigned int*)(EXTR_REG_BASE + 13*4);
	finish = (unsigned int*)(EXTR_REG_BASE + 14*4);
	ctrl = (unsigned int*)(EXTR_REG_BASE + 15*4);
	*readaddr = (unsigned int)rdmem;
	*writeaddr = (unsigned int)wrmem;
	*imgsize = 0x028001E0;
	*readlen = 0x01E003C0;
	printf("imread");
	cv::Mat img = cv::imread("/test.png",cv::IMREAD_GRAYSCALE);
	img.convertTo(img,CV_8U);
	unsigned char* imgtr = rdmem;
	for(int i = 0; i < img.cols; i++)
		for(int j = 0; j < img.rows; j++)
			*imgtr = img.at<uchar>(i,j);
			imgtr++;
	*ctrl = 1;
	while(*finish == 0);
	printf("%d",*finish);
}
