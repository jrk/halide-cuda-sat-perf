Halide performance debugging
============================

This app partitions a 4096x4096 image into 32x32 tiles and computes the summed area table within each tile.

Compilation
-----------
- Makefile provided for both projects
- Edit Makefile.common to set the CUDA include path and Halide base path

Halide version
--------------
- This code runs several kernels - only one computes the summed area table, other kernels are meant to
demonstrate the effect of different Halide update definitions on instruction count and global memory throughput.


CUDA version
------------
- Source code from GPU efficient recursive filtering and summed area table (SIGGRAPH 2011), [Nehab et al.]
