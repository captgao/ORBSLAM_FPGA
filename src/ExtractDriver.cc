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
long long unsigned testimg2[] = {
		0xe502b2ad1f290000,0xc0d7ea4f03230000,0xf7e8c9145abe0000,0x2b39ccf27d840000,0x782c534409e10000,0x8153bf40386c0000,0x87cb676625d60000,0x44c9d6d01fae0000,0x0e12bf6b5d520000,0x5f1e14c4d4900000,0x5033d630cb490000,0x00747eb7fcf10000,0xd49e2d3296f10000,0x610cdc3bf5bb0000,0x8df48ea145e90000,0xbed566223beb0000,0x7bd483f613b30000,0x059fef220da60000,0x15d4579189db0000,0x07a4499d0a3c0000,0x3b5961e11c870000,0x337eff8bdb0c0000,0x8235691fae3e0000,0x1fcf8fda32990000,0x183261b020240000,0x7022cdca9a5e0000,0x92f4d199500d0000,0xdacc1e02ee1c0000,0x64cf9db940060000,0x54d39c7278b70000,0xce90169d36470000,0xb12d7249fdde0000,0x8548722c12b30000,0x3ed3e68049120000,0x698f1d7e324d0000,0x1575f0c5f6c80000,0xf8e684999e430000,0x46d94fd57dbb0000,0x6a1d4ae9498b0000,0x042a7780dca60000,
		0xb4caff6d9f15c596,0x31bea00ee21d6073,0x0e7dd0e3a59a590e,0xaa9f3b6ce2953dd9,0xad8975280cc19716,0xcf8a06a19b9b272f,0xd5418c1eb4e18a67,0xb01b7e2047c05968,0xcafd871d657e76d4,0x27b878cb38e92df7,0xff4f73c22aa8d04a,0xd8684d03469ac24a,0x9df6d03f89a7c9d0,0x1472be41a986cd57,0x4d7b820782c26868,0xf414be8479b5d476,0x7999db0f7a5449fa,0x27cdc21476bf6a16,0x59d34605789a79bb,0x420d4165c2e72511,0x7cf02b1b63d908ad,0x9c448c28b12361ae,0xc13afa6126d14024,0xf8b430c9df551488,0xcda67fc5da90b179,0x8c6670e729383bfe,0x8753f02c6d286a52,0x2033a78e3ed1a5db,0x230b544662d91125,0x64cb8636e06c2843,0xb8a1320896a1c1e5,0xa61095dc12668c3c,0x875eaaf3345ed6f4,0x954c5ba8bf4ea945,0x4cec688d39e10bd3,0xb00313fea63087d8,0x5a4c0bbe3f9c9728,0x8d73e6f289fe8cce,0x4ee6fceb5ed92f0b,0x2d05f571f171f1f5,
		0x92b88c848d88ca99,0x742f98c674816fe7,0xaf7310c3d938d53d,0x8de2ffb18b61cfb1,0xb5519aa7436bd360,0xa47e43306568a1de,0x907dcd4a111295b1,0xd51dcf100862ce80,0x5ed857eea5f95aad,0x6a84c775f654be08,0x40d3506f79d06541,0xfc1f5903bde727e9,0x8001bf2ff7712a67,0x76bebd9eeb17f641,0x02501c6a154807a5,0x4b6b27efb878add5,0x17160310e00da19f,0x6bd62850e192bee4,0x36437f9b60296518,0xb1215dc88f1da69f,0x218389816e86b415,0xdb195f433c29c942,0x7d15b9297b99c000,0x5a184928f4726926,0xea98348a5bdb32fe,0x722b4ef66274344c,0x1e2c60e98a1c09d1,0x822e3c9e8afa2c21,0x8d8be5478f4f4d04,0x71f9dea127370193,0xa80e02815cb88f2f,0x8cdc9848f7b517b3,0xb8bc4231e5b0568f,0x5ef0b26c8795c673,0xd9ca0dcd4a57db53,0x4e0e2ba43bf59d40,0xaf3db69e32dfc843,0xfa6d14de9b80a68a,0xbf94ec81616cd8af,0xb031bba3406d0b7e,
		0xbf0e97322bc3ad94,0xc96a12b9f88fe174,0x190acaf437d62a1d,0x14425704b82ad575,0xd1c40202ed21b2e5,0x3f71b62dd51432dc,0xf8be6c2c7aa95d10,0xf20f08e0a0d40a58,0x08e36e1e5c04e546,0x68c33f014e115ada,0x5101b074fa01dcf2,0xaddbe0f89f18e95b,0xc0ff3a43218d7781,0xc5301c90fcae5da0,0xecb29b0b3cbbeb7f,0x021adfa23673b55c,0x94893a37851c69cb,0x12f85f168e60c51d,0xec65409281ca3a36,0x9422dd08b0206ce9,0x6951ba537dcf9349,0x303cb91cbf5d9874,0xf605e4b5eed60d02,0x833d1b37b12f5755,0xaf4ac03ee445ebd2,0x63b3a637855387ac,0x95be610ee9299a1a,0x41fd1d7212d7df0b,0xe38d92fb07a804f7,0xbd9ee5f28b5968a9,0xdd19c846c6ccb226,0xa1d0bcccec0da223,0x58a1782666ead540,0xdafe2b09e226e65b,0x4f694a7ea3eda4a3,0x31d477f6f355c633,0x38fac37ab94ebcb9,0xce0418d8f4807735,0xfd6759ed90845f88,0xcaed0a5506d98d68,
		0x7c0c62b7b804b1c5,0x70af4f26ae87f45c,0x8c833db6fdb698ef,0x80fd84a2a9308f65,0x3f23ee78492274e5,0xe0a2a0d82bdd1869,0x4d44a80231e422c4,0x473169d151564f30,0x0a5c0ceecadd58de,0x5fb322af80613faf,0x7418d013ec7b749c,0x2ada24212e6e47e2,0x9a307fba453def5e,0x430918591a8c42b6,0x4be13f29395c5436,0x596128de35e72e4b,0x3034c0c6aac4dfbe,0x5a75a9a6be67be00,0xbb77cb2a1cbc3956,0x2b500165652fb62f,0x949affcfcef2e687,0xdcedd9f477ff0c84,0x6c4abd47bdc9676a,0xb563e43657cc1c54,0x5afb6421b30f1b36,0x9d3db8ebd313745f,0x2e4933ddc9905c6c,0x9ce18b4b17a9feda,0x64549aa20a24773c,0x87f9d96a4e119dc5,0x9802c399161e6b0a,0x272c40a832ad6ecb,0xa84e8e12f64b9349,0x48b606c00f41483d,0xee0a61e086e5a705,0x98973734109244f3,0x33c274911328b3ca,0x48793220d23f3e0e,0x489f836e2d6202d7,0xbd7920c784f5b8f2,
		0x08f02c54748d136e,0xaa3f8e4dd7aaeade,0x481ca9496d1760f1,0x0f55c4cd9fb13ec2,0x41ad9d499aae59a1,0xc8514072a70595c3,0xd04fc10e7505250e,0x14c432215a29dc49,0xa385719d30c6f2b5,0x9196afbfcd82afc7,0xe858f28b907f6d80,0xb35e64bea92880bc,0x50d5d0d3a5c0d23e,0x2f88f23987efe5c2,0x601e48044b6a830b,0x908141e1f41257e2,0x2b56d6fd8e4245b9,0x858c4640afe9e9ce,0xe3bb5fa4703fa0b7,0xb7e9091dee8bc4b1,0xe39f9637a331ec11,0x1d6dff0aa64f67b3,0x2025841f2ab10f76,0x2fc8e665258afa64,0x2deb5f740a770269,0x6209c5508bf73ba3,0x280d17956b908f08,0xd319c568d8aecacd,0x501d3e7dd904e415,0x104afc47b09fe128,0x170733ba8bedf91c,0x5d31631d08d6b9eb,0xe701c336d612d1e9,0xe85884d24e262d5a,0xf7ec9234327f22b8,0xc497ab9ed1ec56bf,0xe2d5082381aeeeb7,0xa80da3f677fc3c55,0xe17caa44774503da,0xc8153f39fb01cb5c,
		0xa3fc35489f9d15cf,0xb5590f6eb137023a,0x8e174eeec63ac065,0x038959c3d9ef20ca,0x16a250b51d06a258,0x35bed63a8acce82c,0xbf20f890864ba52d,0x9a07a312fc8cf2e2,0x139dde6bccfff20c,0x3dab277273b6ea60,0xed02551c244c07dc,0x88a7e00f5086d72c,0x202d6cd181576162,0xcaedeb6e83198705,0x6aa7f7dcb225b73c,0xe7b96bedb4367062,0xc62fbb1a71d7dbfa,0xd27d7425a32587c5,0xa5703d3fbd93b199,0x762f36728c72d7b0,0x949fc1ca3bcbe527,0x35fb6c9e8b8ee940,0x51d4777b75ce4468,0x32ae61055ba731c3,0x675bf17529bfe1ab,0xf2570ba8ed3c1dba,0xc34b4c760d69732d,0x26e50b0995288224,0xe829bccab2878dc1,0x96d28dcbef437310,0x9c6a59e765799c9b,0x831e6869aec6c6c4,0x55fc58d14482ce61,0xfb25e47edf15d4f1,0xa94f1bb8e7f957fc,0xf312137477a13dd8,0x8573c071411facbf,0x2effaa9a22f7b04a,0x073f0fb0af6da1d3,0x2b02a6ecaa3e06e6,
		0x8d6fbed37a74f526,0xd33d13ca7e6e7db1,0x9db46de4485dfa8b,0x9bd983c7972977bf,0xfe4245fd14698e4a,0x692aa270137051e3,0x5cf68bc6a4f6e260,0x173afde475bfcf3b,0x9be49198e0c69b22,0xa333c807b43d2ebc,0xc3553907ea2282f2,0xeaf3d95464c9896f,0x4d7893f11740da08,0x5278423a4c4a67bd,0xa9be8af2aecd300b,0x9f49e541b392ba0a,0x40ffaf5f93c4dfc3,0x81897594d95cd627,0x40d6a33b43f9ab3c,0xaa61c7a37e262e6b,0xe9dabfc8d6011207,0x64d75ff166fbd06c,0x3f99d89ddc60467f,0xfc5493553a65ffb3,0x48895a664e26f73c,0x8b68cc32392358e3,0x404112c1f15b491a,0xd5859d6e8e57661f,0x6ab956cbd8edb280,0x5bcfcd354d3e93bf,0x666a7462e50294c7,0xda338cce64bfd295,0x337e8b2382d8caef,0xbf5c6605034f7d42,0x44733eafe5958511,0x674ab0f741f3377c,0x5adddaba95ba2d22,0x1b90874d1470d92e,0xfa27f1f0131b1c01,0x267f961fd1eaa9d1};

namespace ORB_SLAM2{

struct FPGA_ORBResult{
    int batch;
    unsigned int sumrow24;
    unsigned char sumcol[3];
    unsigned char fast;
    unsigned int pixid;
};
struct FPGA_Descriptor{
    unsigned char d[32];
};

void FPGAExtract(unsigned char* mat, unsigned int height, unsigned int width, unsigned int batch){
//    printf("FPGAExtract start \n");
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
    usleep(1);
    *reset=0;

    *toWriteaddr = WR_PADDR;
    *toReadaddr = RD_PADDR;
//    *readlen = (width << 16) + width * 2;
//    *imgsize = (height << 16) + width;
    *readlen = 0x00280050;
    *imgsize = 0x00400028;
    *batchreg = batch;
    *toReadaddr = RD_PADDR;
    printf("readlen %x, imgsize %x\n",*readlen,*imgsize);
 //   printf("0x%x 0x%x %x",*toWriteaddr,*toReadaddr,RD_PADDR);

    memcpy(toWrite,testimg2,64*40);
    syscall(399,(void*)toWrite,409600);
    syscall(400,WR_PADDR,409600);
    syscall(401,WR_PADDR,409600);
    *ctrl = 1;
 //   printf("ctrl %d\n\n",*ctrl);
    while(*ctrl != 0){
	//printf("%x\n",*ctrl);
	usleep(1);
    };
    if(dev_fd)
		close(dev_fd);
    munmap(map_base,4096UL);
    munmap(toWrite, 409600);
    munmap(toRead, 409600);
//    printf("FPGAExtract end\n");
}

void FPGAResult(vector<KeyPoint>& _keypoints, OutputArray& _descriptors, int level){
//    printf("FPGAResult start\n");
	static FILE* fp;
	printf("size %d\n",sizeof(struct FPGA_ORBResult));
    int dev_fd;
	dev_fd = open("/dev/mem",O_RDWR|O_SYNC);
    void* map_base = (unsigned char*) mmap(NULL,4096UL,PROT_READ|PROT_WRITE,
        MAP_SHARED,dev_fd,EXTR_REG_BASE);
    int* heap_outstart = (int*)(map_base + 20);
    *heap_outstart = 1;
   // printf("start wait\n");
    while(*heap_outstart != 0){
	printf("%d\n",*heap_outstart);
//	usleep(1);
    }
    struct FPGA_ORBResult * orb = (struct FPGA_ORBResult*) mmap(NULL,1048576,PROT_READ,MAP_SHARED,dev_fd,ORB_BASE);
    struct FPGA_Descriptor * descstart = (struct FPGA_Descriptor*) mmap(NULL,1048576,PROT_READ,MAP_SHARED,dev_fd,DESC_BASE);
    for(int i =0; i < 1024; i++){
	struct FPGA_ORBResult * orbstart = orb + i;
        if(orbstart->fast != 0){
	    printf("get! %d\n",orbstart->batch);
            int sumrow = orbstart->sumrow24;
            if(sumrow & 0x00800000){
                sumrow = sumrow | 0xff000000;
            }
            else{
                sumrow = sumrow & 0x00ffffff;
            }
            int* tmp = (int*)orbstart -> sumcol;
            int sumcol = *tmp;
            if(sumcol & 0x00800000){
                sumcol = sumcol | 0xff000000;
            }
            else{
                sumcol = sumcol & 0x00ffffff;
            }
		printf("pixid %d\n",orbstart->pixid);
            float angle = fastAtan2((float)sumrow,(float)sumcol);
            _keypoints.push_back(KeyPoint((float)orbstart->pixid,(float)level,(float)orbstart->batch,angle,(float)orbstart->fast));
            if(fp == NULL) fp = fopen("/root/desc.txt","w");
	    struct FPGA_Descriptor * curd = descstart + i;
	    for(int i = 0;i < 32; i++){
		fprintf(fp,"%2x ",curd->d[i]);
	    }
	    fprintf(fp,"\n");
	}
    }
    printf("FPGAResult end\n");
    munmap(orb,1048576);
    munmap(descstart,1048576);
}

void FPGACvtMat(cv::Mat cvmat, unsigned char* mat){
    unsigned char* orgmat = mat;
    for(int i = 0; i < cvmat.rows; i+=8){
        for(int j = 0;j < cvmat.cols; j++){
            for(int k=0;k<8;k++){
                *mat = cvmat.at<uchar>(k+i,j);
                mat++;
            }
        }
    }
    printf("\n\n");
    for(int i = 0; i < 10; i++){
        printf("%2x ",*(orgmat+i));
    }
    printf("\n\n");
}
}
