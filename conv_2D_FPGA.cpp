//===- conv_2D.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


//#include "./fpm/include/fpm/fixed.hpp"
//#include "./fpm/include/fpm/ios.hpp"
//#include "./fpm/include/fpm/math.hpp"
#include "./conv_2D_FPGA.hpp"
#include "./conv.hpp"

//#define float fpm::fixed_16_16
//#define double fpm::fixed_16_16

static float feature_in[W1*H1*CHin_Conv1*CHout_Conv1]={
		#include "./para_BN_W&b_txt/test_image.hpp"
		};

//first layer
static float weight1[CHin_Conv1*K_H1*K_W1*CHout_Conv1]={
		#include "./para_BN_W&b_txt/conv1_weight.hpp"
		};
static float bias1[CHout_Conv1]={
		#include "./para_BN_W&b_txt/conv1_bias.hpp"
		};
//second layer
static float weight2[CHin_Conv2*K_H2*K_W2*CHout_Conv2]={
		#include "./para_BN_W&b_txt/conv2_weight.hpp"
		};
static float bias2[CHout_Conv2]={
		#include "./para_BN_W&b_txt/conv2_bias.hpp"
		};
//third layer
static float weight3[CHin_Conv3*K_H3*K_W3*CHout_Conv3]={
		#include "./para_BN_W&b_txt/conv3_weight.hpp"
		};
static float bias3[CHout_Conv3]={
		#include "./para_BN_W&b_txt/conv3_bias.hpp"
		};
//fourth layer
static float weight4[CHin_Conv4*K_H4*K_W4*CHout_Conv4]={
		#include "./para_BN_W&b_txt/conv4_weight.hpp"
		};
static float bias4[CHout_Conv4]={
		#include "./para_BN_W&b_txt/conv4_bias.hpp"
		};

float conv1_out[OUT_H1*OUT_W1*CHout_Conv1];
float conv2_out[OUT_H2*OUT_W2*CHout_Conv2];
float conv3_out[OUT_H3*OUT_W3*CHout_Conv3];
float conv4_out[OUT_H4*OUT_W4*CHout_Conv4];

int main()
{
    //Conv1
    RunConv(CHin_Conv1,H1,W1,CHout_Conv1,//CHin,Hin,Win,CHout
            K_H1,K_W1,STRIDE_H1,STRIDE_W1,PADDING,RELU_EN,//Kx,Ky,Sx,Sy,mode,relu_en
            feature_in,weight1,bias1,conv1_out);//feature_in,W,bias,feature_out
    // Print conv1_out
    printf("Conv1 Output:\n");
    for (int i = 0; i < OUT_H1 * OUT_W1 * CHout_Conv1; i++) {
        printf("%f ", conv1_out[i]);
    }
    printf("\n");
    
    //Conv2
    RunConv(CHin_Conv2,H2,W2,CHout_Conv2,//CHin,Hin,Win,CHout
            K_H2,K_W2,STRIDE_H2,STRIDE_W2,PADDING,RELU_EN,//Kx,Ky,Sx,Sy,mode,relu_en
            conv1_out,weight2,bias2,conv2_out);//feature_in,W,bias,feature_out
    // Print conv2_out
    printf("Conv2 Output:\n");
    for (int i = 0; i < OUT_H2 * OUT_W2 * CHout_Conv2; i++) {
        printf("%f ", conv2_out[i]);
    }
    printf("\n");
    
    //Conv3
    RunConv(CHin_Conv3,H3,W3,CHout_Conv3,//CHin,Hin,Win,CHout
            K_H3,K_W3,STRIDE_H3,STRIDE_W3,PADDING,RELU_EN,//Kx,Ky,Sx,Sy,mode,relu_en
            conv2_out,weight3,bias3,conv3_out);//feature_in,W,bias,feature_out
    // Print conv3_out
    printf("Conv3 Output:\n");
    for (int i = 0; i < OUT_H3 * OUT_W3 * CHout_Conv3; i++) {
        printf("%f ", conv3_out[i]);
    }
    printf("\n");
    
    //Conv4
    RunConv(CHin_Conv4,H4,W4,CHout_Conv4,//CHin,Hin,Win,CHout
            K_H4,K_W4,STRIDE_H4,STRIDE_W4,PADDING,0,//Kx,Ky,Sx,Sy,mode,relu_en
            conv3_out,weight4,bias4,conv4_out);//feature_in,W,bias,feature_out
    // Print conv4_out
    printf("Conv4 Output:\n");
    for (int i = 0; i < OUT_H4 * OUT_W4 * CHout_Conv4; i++) {
        printf("%f ", conv4_out[i]);
    }
    printf("\n");
    

    float max=-10000;
    int num=0;
    for(int m=0;m<10;m++)
    {
        if(conv4_out[m]>max)
        {
            max=conv4_out[m];
            num=m;
        }
    }
    printf("predicted=%d, label=7\r\n",num);
    return 0;
}


