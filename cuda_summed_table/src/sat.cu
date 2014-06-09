#include <defs.h>
#include <symbol.h>
#include <gpudefs.h>
#include <dvector.h>
#include <gpuconsts.cuh>

__global__ __launch_bounds__( WS * SOW, MBO )
void algSAT_stage4( float *g_out, const float *g_in) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float s_block[ WS ][ WS+1 ];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

	g_in += (row0+ty)*c_width+col;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        **bdata = *g_in;
        bdata += SOW;
        g_in += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        **bdata = *g_in;
    }

	__syncthreads();

	if( ty == 0 ) {

        {   // calculate y -----------------------
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[0][tx];

            float prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                **bdata = prev = **bdata + prev;
        }

        {   // calculate x -----------------------
            float *bdata = s_block[tx];

            float prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                *bdata = prev = *bdata + prev;
        }

	}

	__syncthreads();

    bdata = (float (*)[WS+1]) &s_block[ty][tx];

	g_out += (row0+ty)*c_width+col;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        *g_out = **bdata;
        bdata += SOW;
        g_out += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        *g_out = **bdata;
    }

}

//-- Host ---------------------------------------------------------------------

__host__
void calc_borders( int& left,
                   int& top,
                   int& right,
                   int& bottom,
                   const int& w,
                   const int& h,
                   const int& extb ) {

    left = extb*WS;
    top = extb*WS;

    if( extb > 0 ) {

        right = (extb+1)*WS-(w%WS);
        bottom = (extb+1)*WS-(h%WS);

    } else {

        right = WS-(w%WS);
        if( right == WS ) right = 0;
        bottom = WS-(h%WS);
        if( bottom == WS ) bottom = 0;

    }

}

__host__
bool extend( const int& w,
             const int& h,
             const int& extb ) {
    return (w%32>0 or h%32>0 or extb>0);
}

__host__
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h ) {

    algs.width = w;
    algs.height = h;
    algs.m_size = (w+WS-1)/WS;
    algs.n_size = (h+WS-1)/WS;
    algs.last_m = algs.m_size-1;
    algs.last_n = algs.n_size-1;
    algs.border = 0;
    algs.carry_width = algs.m_size*WS;
    algs.carry_height = algs.n_size*WS;
    algs.carry_height = h;
    algs.inv_width = 1.f/(float)w;
    algs.inv_height = 1.f/(float)h;

}

__host__
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h,
                     const int& extb ) {

    int bleft, btop, bright, bbottom;
    calc_borders( bleft, btop, bright, bbottom, w, h, extb );

    algs.width = w;
    algs.height = h;
    algs.m_size = (w+bleft+bright+WS-1)/WS;
    algs.n_size = (h+btop+bbottom+WS-1)/WS;
    algs.last_m = (bleft+w-1)/WS;
    algs.last_n = (btop+h-1)/WS;
    algs.border = extb;
    algs.carry_width = algs.m_size*WS;
    algs.carry_height = algs.n_size*WS;
    algs.inv_width = 1.f/(float)w;
    algs.inv_height = 1.f/(float)h;

}


__host__
void prepare_algSAT( alg_setup& algs,
                     dvector<float>& d_inout,
                     const float *h_in,
                     const int& w,
                     const int& h )
{
    algs.width = w;
    algs.height = h;

    if( w % 32 > 0 ) algs.width += (32 - (w % 32));
    if( h % 32 > 0 ) algs.height += (32 - (h % 32));

    calc_alg_setup( algs, algs.width, algs.height );
    up_alg_setup( algs );

    d_inout.copy_from( h_in, w, h, algs.width, algs.height );
}

__host__
void algSAT( dvector<float>& d_out,
             const dvector<float>& d_in,
             const alg_setup& algs ) {

	const int nWm = (algs.width+MTS-1)/MTS, nHm = (algs.height+MTS-1)/MTS;
    const dim3 cg_img( algs.m_size, algs.n_size );
    const dim3 cg_ybar( nWm, 1 );
    const dim3 cg_vhat( 1, nHm );

    algSAT_stage4<<< cg_img, dim3(WS, SOW) >>>( d_out, d_in);
}
