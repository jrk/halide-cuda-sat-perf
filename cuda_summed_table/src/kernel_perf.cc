#include <iostream>
#include <cstdlib>
#include <gpudefs.h>
#include <dvector.h>

extern void algSAT_stage4( float *g_out, const float *g_in);

extern void prepare_algSAT( alg_setup& algs,
                     dvector<float>& d_inout,
                     const float *h_in,
                     const int& w,
                     const int& h );

extern void algSAT( dvector<float>& d_out,
             const dvector<float>& d_in,
             const alg_setup& algs );


int main(int argc, char *argv[]) {

    const int in_w = 4096, in_h = 4096;

    std::cout << "Generating random input image (" << in_w << "x" << in_h << ") ... " << std::endl;

    float *in_gpu = new float[in_w*in_h];

    for (int i = 0; i < in_w*in_h; ++i)
        in_gpu[i] = rand() % 10;

    std::cout << "done!\n[sat3] Configuring the GPU to run ... " << std::endl;

    alg_setup algs;
    dvector<float> d_in_gpu;

    prepare_algSAT( algs, d_in_gpu, in_gpu, in_w, in_h );

    dvector<float> d_out_gpu( algs.width, algs.height );

    std::cout << "Computing summed-area table in the GPU ... " << std::endl;

    algSAT( d_out_gpu, d_in_gpu, algs );

    d_out_gpu.copy_to( in_gpu, algs.width, algs.height, in_w, in_h );

    delete [] in_gpu;

    return 0;
}
