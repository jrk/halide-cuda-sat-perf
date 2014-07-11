// Code to divide the image into 2D tile and compute a summed area table
// within each tile in parallel

#include <iostream>
#include <Halide.h>

#include <sys/time.h>

static double current_time() {
    static bool first_call = true;
    static timeval reference_time;
    if (first_call) {
        first_call = false;
        gettimeofday(&reference_time, NULL);
        return 0.0;
    } else {
        timeval t;
        gettimeofday(&t, NULL);
        return ((t.tv_sec - reference_time.tv_sec)*1000.0 +
                (t.tv_usec - reference_time.tv_usec)/1000.0);
    }
}
static const int timing_runs = 4;

#define TIME(stmt)  double __minT = 1e10; for (int i = 0; i < timing_runs; i++) { double start = current_time(); (stmt); double end = current_time(); double elapsed = end-start; __minT = (elapsed < __minT) ? elapsed : __minT; } __minT
#define GET_TIME    __minT

using namespace Halide;

// image dimensions and tile width
static const int width = 4096;
static const int height= 4096;
static const int tile  = 32;

// different version of the intra tile computation
// see each function for details
void version_0(Func I);
void version_1(Func I);
void version_2(Func I);
void version_3(Func I);
void version_4(Func I);

// apply the same schedule on all the versions
void apply_schedule(Func& S, Func& SI);

// Global vars - ugly but will do for now
Var x("x")  , y("y");
Var xi("xi"), yi("yi");
Var xo("xo"), yo("yo");
RDom rxi(1, tile-1, "rxi");
RDom ryi(1, tile-1, "ryi");

int main(int argc, char **argv)
{
    Image<float> in(width,height);      // input image
    for (int j=0; j<height; j++) {
        for (int i=0; i<width; i++) {
            in(i,j) = float(rand() % 10);
        }
    }

    Func I("Input");
    I(x,y) = in(x,y);

    version_0(I);    // each creates the Halide Funcs,
    version_1(I);    // applies the same schedule
    version_2(I);    // and realizes the result on 4096 x 4096 image
    version_3(I);
    version_4(I);

    return 0;
}


void apply_schedule(Func& S, Func& SI) {
    Target target = get_jit_target_from_environment();

    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");

        SI.compute_at(S, Var("__block_id_x"));
        SI.split(yi, yi, t, 6).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
        SI.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        SI.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

        S.compute_root();
        S.reorder_storage(y, x);
        S.split(x, xo,xi, tile).split(y, yo,yi, tile);
        S.split(yi, yi, t, 6).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
        S.gpu_blocks(xo,yo);
        S.bound(x, 0, width).bound(y, 0, height);
    } else {
        std::cerr << "Error: Set HL_JIT_TARGET=cuda or cuda-gpu_debug" << std::endl;
        exit(-1);
    }
}

void version_0(Func I) {
    Func S ("S_version_0");
    Func SI("SI_version_0");

    SI(xi ,xo,yi ,yo) = I(xo*tile+xi, yo*tile+yi);
    SI(rxi,xo,yi ,yo) = SI(rxi,xo,yi ,yo) + SI(rxi-1, xo, yi, yo);
    SI(xi ,xo,ryi,yo) = SI(xi ,xo,ryi,yo) + SI(xi, xo, ryi-1, yo);

    S(x,y) = SI(x%tile, x/tile, y%tile, y/tile);    // final image

    apply_schedule(S, SI);

    TIME(S.realize(width,height));
    double t = GET_TIME;
    std::cerr << "version_0 took " << t << " ms\n";
}

void version_1(Func I) {
    Func S ("S_version_1");
    Func SI("SI_version_1");

    SI(xi ,xo,yi ,yo) = I(xo*tile+xi, yo*tile+yi);
    SI(rxi,xo,yi ,yo) = SI(rxi,xo,yi ,yo);
    SI(xi ,xo,ryi,yo) = SI(xi ,xo,ryi,yo);

    S(x,y) = SI(x%tile, x/tile, y%tile, y/tile);    // final image

    apply_schedule(S, SI);

    TIME(S.realize(width,height));
    double t = GET_TIME;
    std::cerr << "version_1 took " << t << " ms\n";
}

void version_2(Func I) {
    Func S ("S_version_2");
    Func SI("SI_version_2");

    SI(xi ,xo,yi ,yo) = I(xo*tile+xi, yo*tile+yi);
    SI(rxi,xo,yi ,yo) = SI(rxi-1, xo, yi, yo);
    SI(xi ,xo,ryi,yo) = SI(xi, xo, ryi-1, yo);

    S(x,y) = SI(x%tile, x/tile, y%tile, y/tile);    // final image

    apply_schedule(S, SI);

    TIME(S.realize(width,height));
    double t = GET_TIME;
    std::cerr << "version_2 took " << t << " ms\n";
}

void version_3(Func I) {
    Func S ("S_version_3");
    Func SI("SI_version_3");

    SI(xi ,xo,yi ,yo) = I(xo*tile+xi, yo*tile+yi);
    SI(rxi,xo,yi ,yo) = SI(rxi,xo,yi ,yo) + SI(rxi, xo, yi, yo);
    SI(xi ,xo,ryi,yo) = SI(xi ,xo,ryi,yo) + SI(xi, xo, ryi, yo);

    S(x,y) = SI(x%tile, x/tile, y%tile, y/tile);    // final image

    apply_schedule(S, SI);

    TIME(S.realize(width,height));
    double t = GET_TIME;
    std::cerr << "version_3 took " << t << " ms\n";
}

void version_4(Func I) {
    Func S ("S_version_4");
    Func SI("SI_version_4");

    SI(xi ,xo,yi ,yo) = I(xo*tile+xi, yo*tile+yi);
    SI(rxi,xo,yi ,yo) = 0.0f;
    SI(xi ,xo,ryi,yo) = 0.0f;

    S(x,y) = SI(x%tile, x/tile, y%tile, y/tile);    // final image

    apply_schedule(S, SI);

    TIME(S.realize(width,height));
    double t = GET_TIME;
    std::cerr << "version_4 took " << t << " ms\n";
}
