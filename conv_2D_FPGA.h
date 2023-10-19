
//first layer
#define H1 28
#define W1 28
#define K_H1 7
#define K_W1 7
#define STRIDE_H1 2
#define STRIDE_W1 2
#define PAD_H1 0
#define PAD_W1 0

#define CHin_Conv1 1
#define CHout_Conv1 3

#define OUT_H1 ((H1 - K_H1 + 2 * PAD_H1 + STRIDE_H1) / STRIDE_H1)
#define OUT_W1 ((W1 - K_W1 + 2 * PAD_W1 + STRIDE_W1) / STRIDE_W1)

//second layer
#define H2 11
#define W2 11
#define K_H2 5
#define K_W2 5
#define STRIDE_H2 1
#define STRIDE_W2 1
#define PAD_H2 0
#define PAD_W2 0

#define CHin_Conv2 3
#define CHout_Conv2 5

#define OUT_H2 ((H2 - K_H2 + 2 * PAD_H2 + STRIDE_H2) / STRIDE_H2)
#define OUT_W2 ((W2 - K_W2 + 2 * PAD_W2 + STRIDE_W2) / STRIDE_W2)

//third layer
#define H3 7
#define W3 7
#define K_H3 5
#define K_W3 5
#define STRIDE_H3 1
#define STRIDE_W3 1
#define PAD_H3 0
#define PAD_W3 0

#define CHin_Conv3 5
#define CHout_Conv3 7

#define OUT_H3 ((H3 - K_H3 + 2 * PAD_H3 + STRIDE_H3) / STRIDE_H3)
#define OUT_W3 ((W3 - K_W3 + 2 * PAD_W3 + STRIDE_W3) / STRIDE_W3)

//fourth layer
#define H4 3
#define W4 3
#define K_H4 3
#define K_W4 3
#define STRIDE_H4 1
#define STRIDE_W4 1
#define PAD_H4 0
#define PAD_W4 0

#define CHin_Conv4 7
#define CHout_Conv4 10

#define OUT_H4 ((H4 - K_H4 + 2 * PAD_H4 + STRIDE_H4) / STRIDE_H4)
#define OUT_W4 ((W4 - K_W4 + 2 * PAD_W4 + STRIDE_W4) / STRIDE_W4)
