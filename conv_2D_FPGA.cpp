//===- conv_2D.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: clang++ %syclops-clang-device-only-flags %s -o %t.ll
// RUN: syclops %t.ll -emit-mlir -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=MLIR
// RUN: syclops %t.ll -emit-akg -o %t.txt

#include "../test_common.hpp"
#include <CL/sycl.hpp>
#include "/home/zzy666/Desktop/SYCLops/SYCLops/test/dnn_kernels/verify_conv/fpm/include/fpm/fixed.hpp"
#include "/home/zzy666/Desktop/SYCLops/SYCLops/test/dnn_kernels/verify_conv/fpm/include/fpm/ios.hpp"
#include "/home/zzy666/Desktop/SYCLops/SYCLops/test/dnn_kernels/verify_conv/fpm/include/fpm/math.hpp"
#include "conv_2D_FPGA.hpp"

static float feature_in[W1*H1*CHin_Conv1*CHout_Conv1]={
		#include "./parameters/test_image.hpp"
		};
//first layer
static float weight1[CHin_Conv1*K_H1*K_W1*CHout_Conv1]={
		#include "./parameters/conv1_weight.hpp"
		};
static float bias1[CHout_Conv1]={
		#include "./parameters/conv1_bias.hpp"
		};
//second layer
static float weight2[CHin_Conv2*K_H2*K_W2*CHout_Conv2]={
		#include "./parameters/conv2_weight.hpp"
		};
static float bias2[CHout_Conv2]={
		#include "./parameters/conv2_bias.hpp"
		};
//third layer
static float weight3[CHin_Conv3*K_H3*K_W3*CHout_Conv3]={
		#include "./parameters/conv3_weight.hpp"
		};
static float bias3[CHout_Conv3]={
		#include "./parameters/conv3_bias.hpp"
		};
//fourth layer
static float weight4[CHin_Conv4*K_H4*K_W4*CHout_Conv4]={
		#include "./parameters/conv4_weight.hpp"
		};
static float bias4[CHout_Conv4]={
		#include "./parameters/conv4_bias.hpp"
		};
		
#define float fpm::fixed_16_16
#define double fpm::fixed_16_16

using namespace ::sycl;

//layer1
using _Array_in1 = Array<float, H1 * W1 *CHin_Conv1>;
using _Array_kern1 = Array<float, CHin_Conv1 * K_H1 * K_W1 * CHout_Conv1>;
using _Array_bias1 = Array<float, CHout_Conv1>;
using _Array_out1 = Array<float, OUT_H1 * OUT_W1 * CHout_Conv1>;
//layer2
using _Array_in2 = Array<float, H2 * W2 *CHin_Conv2>;
using _Array_kern2 = Array<float, CHin_Conv2 * K_H2 * K_W2 * CHout_Conv2>;
using _Array_bias2 = Array<float, CHout_Conv2>;
using _Array_out2 = Array<float, OUT_H2 * OUT_W2 * CHout_Conv2>;
//layer3
using _Array_in3 = Array<float, H3 * W3 *CHin_Conv3>;
using _Array_kern3 = Array<float, CHin_Conv3 * K_H3 * K_W3 * CHout_Conv3>;
using _Array_bias3 = Array<float, CHout_Conv3>;
using _Array_out3 = Array<float, OUT_H3 * OUT_W3 * CHout_Conv3>;
//layer4
using _Array_in4 = Array<float, H4 * W4 *CHin_Conv4>;
using _Array_kern4 = Array<float, CHin_Conv4 * K_H4 * K_W4 * CHout_Conv4>;
using _Array_bias4 = Array<float, CHout_Conv4>;
using _Array_out4 = Array<float, OUT_H4 * OUT_W4 * CHout_Conv4>;

int main() {

  float feature_in_fixed[sizeof(feature_in)/4];
  for(int i=0;i<sizeof(feature_in)/4;i++){
  	float temp{feature_in[i]};
  	feature_in_fixed[i]=temp;
}
  auto &feature_in=feature_in_fixed;
  
  //layer1
  float weight_fixed1[sizeof(weight1)/4];
  for(int i=0;i<sizeof(weight1)/4;i++){
  	float temp{weight1[i]};
  	weight_fixed1[i]=temp;
}
  auto &weight1=weight_fixed1;
  
  float bias_fixed1[sizeof(bias1)/4];
  for(int i=0;i<sizeof(bias1)/4;i++){
  	float temp{bias1[i]};
  	bias_fixed1[i]=temp;
}
  auto &bias1=bias_fixed1;
  
  //layer2
  float weight_fixed2[sizeof(weight2)/4];
  for(int i=0;i<sizeof(weight2)/4;i++){
  	float temp{weight2[i]};
  	weight_fixed2[i]=temp;
}
  auto &weight2=weight_fixed2;
  
  float bias_fixed2[sizeof(bias2)/4];
  for(int i=0;i<sizeof(bias2)/4;i++){
  	float temp{bias2[i]};
  	bias_fixed2[i]=temp;
}
  auto &bias2=bias_fixed2;
  
  //layer3
  float weight_fixed3[sizeof(weight3)/4];
  for(int i=0;i<sizeof(weight3)/4;i++){
  	float temp{weight3[i]};
  	weight_fixed3[i]=temp;
}
  auto &weight3=weight_fixed3;
  
  float bias_fixed3[sizeof(bias3)/4];
  for(int i=0;i<sizeof(bias3)/4;i++){
  	float temp{bias3[i]};
  	bias_fixed3[i]=temp;
}
  auto &bias3=bias_fixed3;
  
  //layer4
    float weight_fixed4[sizeof(weight4)/4];
  for(int i=0;i<sizeof(weight4)/4;i++){
  	float temp{weight4[i]};
  	weight_fixed4[i]=temp;
}
  auto &weight4=weight_fixed4;

  float bias_fixed4[sizeof(bias4)/4];
  for(int i=0;i<sizeof(bias4)/4;i++){
  	float temp{bias4[i]};
  	bias_fixed4[i]=temp;
}
  auto &bias4=bias_fixed4;

  queue deviceQueue(default_selector_v);
  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();
  
//  auto* IN_acc = (_Array_in *)(aligned_alloc( 4096, sizeof(_Array_in) ));
//  auto* KERNEL_acc = (_Array_kern *)(aligned_alloc( 4096, sizeof(_Array_kern) ));
//  auto* OUT_acc = (_Array_out *)(aligned_alloc( 4096, sizeof(_Array_out) ));

  //layer1
  auto IN_acc1 = (_Array_in1 *)malloc_shared(sizeof(_Array_in1), dev, ctx);
  auto KERNEL_acc1 = (_Array_kern1 *)malloc_shared(sizeof(_Array_kern1), dev, ctx);
  auto BIAS_acc1 = (_Array_bias1 *)malloc_shared(sizeof(_Array_bias1), dev, ctx);
  
  //layer2
  auto OUT_acc1 = (_Array_in2 *)malloc_shared(sizeof(_Array_in2), dev, ctx);
  auto KERNEL_acc2 = (_Array_kern2 *)malloc_shared(sizeof(_Array_kern2), dev, ctx);
  auto BIAS_acc2 = (_Array_bias2 *)malloc_shared(sizeof(_Array_bias2), dev, ctx);
  
  //layer3
  auto OUT_acc2 = (_Array_in3 *)malloc_shared(sizeof(_Array_in3), dev, ctx);
  auto KERNEL_acc3 = (_Array_kern3 *)malloc_shared(sizeof(_Array_kern3), dev, ctx);
  auto BIAS_acc3 = (_Array_bias3 *)malloc_shared(sizeof(_Array_bias3), dev, ctx);
  
  //layer4
  auto OUT_acc3 = (_Array_in4*)malloc_shared(sizeof(_Array_in4), dev, ctx);
  auto KERNEL_acc4 = (_Array_kern4 *)malloc_shared(sizeof(_Array_kern4), dev, ctx);
  auto BIAS_acc4 = (_Array_bias4 *)malloc_shared(sizeof(_Array_bias4), dev, ctx);
  auto OUT_acc4 = (_Array_out4*)malloc_shared(sizeof(_Array_out4), dev, ctx);
  
  std::cout << "================IMAGE INPUT=============" << std::endl;
  for (int i = 0; i < W1*H1*CHin_Conv1; i++){
  	 (*IN_acc1)[i] = feature_in[i];
  	 std::cout << (*IN_acc1)[i] << " ";
  	 }
  std::cout<<""<<std::endl;
  	 
  std::cout << "================IMPORTED WEIGHT1=============" << std::endl;
  for (int i = 0; i < K_W1*K_H1*CHin_Conv1*CHout_Conv1; i++){
  	(*KERNEL_acc1)[i] = weight1[i];
  	std::cout << (*KERNEL_acc1)[i] << " ";
  	}
  std::cout<<""<<std::endl;
  std::cout << "================IMPORTED BIAS1=============" << std::endl;
  for (int i = 0; i < CHout_Conv1; i++){
  	(*BIAS_acc1)[i] = bias1[i];
  	std::cout << (*BIAS_acc1)[i] << " ";
  	}
  std::cout<<""<<std::endl;
  	
  std::cout << "================IMPORTED WEIGHT2=============" << std::endl;	
  for (int i = 0; i < K_W2*K_H2*CHin_Conv2*CHout_Conv2; i++){
  	(*KERNEL_acc2)[i] = weight2[i];
  	std::cout << (*KERNEL_acc2)[i] << " ";
  	}
  std::cout<<""<<std::endl;
  std::cout << "================IMPORTED BIAS2=============" << std::endl;
  for (int i = 0; i < CHout_Conv2; i++){
  	(*BIAS_acc2)[i] = bias2[i];
  	std::cout << (*BIAS_acc2)[i] << " ";
  	}
  std::cout<<""<<std::endl;
  	
  std::cout << "================IMPORTED WEIGHT3=============" << std::endl;	
  for (int i = 0; i < K_W3*K_H3*CHin_Conv3*CHout_Conv3; i++){
	(*KERNEL_acc3)[i] = weight3[i];
	std::cout << (*KERNEL_acc3)[i] << " ";
	}
  std::cout<<""<<std::endl;
  std::cout << "================IMPORTED BIAS3=============" << std::endl;
  for (int i = 0; i < CHout_Conv3; i++){
  	(*BIAS_acc3)[i] = bias3[i];
  	std::cout << (*BIAS_acc3)[i] << " ";
  	}
  std::cout<<""<<std::endl;
  	
  std::cout << "================IMPORTED WEIGHT4=============" << std::endl;
  for (int i = 0; i < K_W4*K_H4*CHin_Conv4*CHout_Conv4; i++){
	(*KERNEL_acc4)[i] = weight4[i];
	std::cout << (*KERNEL_acc4)[i] << " ";
	}
  std::cout<<""<<std::endl;
  std::cout << "================IMPORTED BIAS4=============" << std::endl;
  for (int i = 0; i < CHout_Conv4; i++){
  	(*BIAS_acc4)[i] = bias4[i];
  	std::cout << (*BIAS_acc4)[i] << " ";
  	}
  std::cout<<""<<std::endl;
  
  //deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      //================Layer1==============
      for(int cout=0;cout<CHout_Conv1;cout++)
      	for(int i=0;i<OUT_H1;i++)
          for(int j=0;j<OUT_W1;j++)
          {
	       float sum{0};
	       for(int ii=0;ii<K_H1;ii++)
	         for(int jj=0;jj<K_W1;jj++)
		   {
			int h=i*STRIDE_H1+ii;
			int w=j*STRIDE_W1+jj;
			if(h>=0 && w>=0 && h<H1 && w<W1)
			{
				for(int cin=0;cin<CHin_Conv1;cin++)
				{
				float tp;
				float A=(*IN_acc1)[h*CHin_Conv1*W1+w*CHin_Conv1+cin];
				float B=(*KERNEL_acc1)[ii*K_W1*CHin_Conv1*CHout_Conv1+jj*CHin_Conv1*CHout_Conv1+cin*CHout_Conv1+cout];
				tp=(A*B);
				sum+=tp;
				}
			}
		  }
		//sum=scale[cout]*sum+shift[cout];//Batch normalization
		sum=sum+(*BIAS_acc1)[cout];
		sum=(sum>float(0))?sum:float(0);//ReLu
		(*OUT_acc1)[i*OUT_W1*CHout_Conv1+j*CHout_Conv1+cout]=sum;
          }
          
     //================Layer2==============
   for(int cout=0;cout<CHout_Conv2;cout++)
      	for(int i=0;i<OUT_H2;i++)
          for(int j=0;j<OUT_W2;j++)
          {
	       float sum{0};
	       for(int ii=0;ii<K_H2;ii++)
	         for(int jj=0;jj<K_W2;jj++)
		   {
			int h=i*STRIDE_H2+ii;
			int w=j*STRIDE_W2+jj;
			if(h>=0 && w>=0 && h<H2 && w<W2)
			{
				for(int cin=0;cin<CHin_Conv2;cin++)
				{
				float tp;
				float A=(*OUT_acc1)[h*CHin_Conv2*W2+w*CHin_Conv2+cin];
				float B=(*KERNEL_acc2)[ii*K_W2*CHin_Conv2*CHout_Conv2+jj*CHin_Conv2*CHout_Conv2+cin*CHout_Conv2+cout];
				tp=(A*B);
				sum+=tp;
				}
			}
		  }
		//sum=scale[cout]*sum+shift[cout];//Batch normalization
		sum=sum+(*BIAS_acc2)[cout];
		sum=(sum>float(0))?sum:float(0);//ReLu
		(*OUT_acc2)[i*OUT_W2*CHout_Conv2+j*CHout_Conv2+cout]=sum;
          }
     
     //================Layer3==============
   for(int cout=0;cout<CHout_Conv3;cout++)
      	for(int i=0;i<OUT_H3;i++)
          for(int j=0;j<OUT_W3;j++)
          {
	       float sum{0};
	       for(int ii=0;ii<K_H3;ii++)
	         for(int jj=0;jj<K_W3;jj++)
		   {
			int h=i*STRIDE_H3+ii;
			int w=j*STRIDE_W3+jj;
			if(h>=0 && w>=0 && h<H3 && w<W3)
			{
				for(int cin=0;cin<CHin_Conv3;cin++)
				{
				float tp;
				float A=(*OUT_acc2)[h*CHin_Conv3*W3+w*CHin_Conv3+cin];
				float B=(*KERNEL_acc3)[ii*K_W3*CHin_Conv3*CHout_Conv3+jj*CHin_Conv3*CHout_Conv3+cin*CHout_Conv3+cout];
				tp=(A*B);
				sum+=tp;
				}
			}
		  }
		//sum=scale[cout]*sum+shift[cout];//Batch normalization
		sum=sum+(*BIAS_acc3)[cout];
		sum=(sum>float(0))?sum:float(0);//ReLu
		(*OUT_acc3)[i*OUT_W3*CHout_Conv3+j*CHout_Conv3+cout]=sum;
          }
     //================Layer4==============
   for(int cout=0;cout<CHout_Conv4;cout++)
      	for(int i=0;i<OUT_H4;i++)
          for(int j=0;j<OUT_W4;j++)
          {
	       float sum{0};
	       for(int ii=0;ii<K_H4;ii++)
	         for(int jj=0;jj<K_W4;jj++)
		   {
			int h=i*STRIDE_H4+ii;
			int w=j*STRIDE_W4+jj;
			if(h>=0 && w>=0 && h<H4 && w<W4)
			{
				for(int cin=0;cin<CHin_Conv4;cin++)
				{
				float tp;
				float A=(*OUT_acc3)[h*CHin_Conv4*W4+w*CHin_Conv4+cin];
				float B=(*KERNEL_acc4)[ii*K_W2*CHin_Conv4*CHout_Conv4+jj*CHin_Conv4*CHout_Conv4+cin*CHout_Conv4+cout];
				tp=(A*B);
				sum+=tp;
				}
			}
		  }
		//sum=scale[cout]*sum+shift[cout];//Batch normalization
		sum=sum+(*BIAS_acc4)[cout];
		sum=(sum>float(0))?sum:float(0);//ReLu
		(*OUT_acc4)[i*OUT_W4*CHout_Conv4+j*CHout_Conv4+cout]=sum;
          }
     
     
    };
    kern();//cgh.single_task<class conv_2D>(kern);
  //});

  deviceQueue.wait();
  printf("Queue wait done \n");
  
  std::cout << "================RESULT FROM layer 1=============" << std::endl;
  for(int cout=0;cout<CHout_Conv1;cout++)
    for(int i=0;i<OUT_H1;i++)
      for(int j=0;j<OUT_W1;j++)
        {
	std::cout<<"["<<j<<"]"<<"["<<i<<"]"<<"["<<cout<<"]"<<"element   "<<(*OUT_acc1)[i*OUT_W1*CHout_Conv1+j*CHout_Conv1+cout]<<"\n";
	}
  std::cout << "==================================" << std::endl;
  
  std::cout << "================RESULT FROM layer1 one-dimension=============" << std::endl;
  for(int i=0;i<CHout_Conv1*OUT_W1*OUT_H1;i++)
        {
	std::cout<<""<<(*OUT_acc1)[i]<<"\n";
	}
  std::cout << "==================================" << std::endl;
  
  std::cout << "================RESULT FROM layer 2=============" << std::endl;
  for(int cout=0;cout<CHout_Conv2;cout++)
    for(int i=0;i<OUT_H2;i++)
      for(int j=0;j<OUT_W2;j++)
        {
	std::cout<<"["<<j<<"]"<<"["<<i<<"]"<<"["<<cout<<"]"<<"element   "<<(*OUT_acc2)[i*OUT_W2*CHout_Conv2+j*CHout_Conv2+cout]<<"\n";
	}
  std::cout << "==================================" << std::endl;
  
  std::cout << "================RESULT FROM layer 3=============" << std::endl;
  for(int cout=0;cout<CHout_Conv3;cout++)
    for(int i=0;i<OUT_H3;i++)
      for(int j=0;j<OUT_W3;j++)
        {
	std::cout<<"["<<j<<"]"<<"["<<i<<"]"<<"["<<cout<<"]"<<"element   "<<(*OUT_acc3)[i*OUT_W3*CHout_Conv3+j*CHout_Conv3+cout]<<"\n";
	}
  std::cout << "==================================" << std::endl;
  
  std::cout << "================RESULT FROM layer 4=============" << std::endl;
  for(int cout=0;cout<CHout_Conv4;cout++)
    for(int i=0;i<OUT_H4;i++)
      for(int j=0;j<OUT_W4;j++)
        {
	std::cout<<"["<<j<<"]"<<"["<<i<<"]"<<"["<<cout<<"]"<<"element   "<<(*OUT_acc4)[i*OUT_W4*CHout_Conv4+j*CHout_Conv4+cout]<<"\n";
	}
  std::cout << "==================================" << std::endl;
  
  sycl::free(IN_acc1, deviceQueue);
  sycl::free(KERNEL_acc1, deviceQueue);
  sycl::free(BIAS_acc1, deviceQueue);
  sycl::free(OUT_acc1, deviceQueue);
  
  sycl::free(KERNEL_acc2, deviceQueue);
  sycl::free(BIAS_acc2, deviceQueue);
  sycl::free(OUT_acc2, deviceQueue);
  
  sycl::free(KERNEL_acc3, deviceQueue);
  sycl::free(BIAS_acc3, deviceQueue);
  sycl::free(OUT_acc3, deviceQueue);
  
  sycl::free(KERNEL_acc4, deviceQueue);
  sycl::free(BIAS_acc4, deviceQueue);
  sycl::free(OUT_acc4, deviceQueue);

  return 0;
}


