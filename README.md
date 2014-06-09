Halide performance debugging
============================

Comparision between Halide and CUDA version of an app that partitions a 4096x4096 image into 32x32 tiles and computes the summed
area table within each tile.

**Halide version**: runs several kernels - only version 0 computes the summed area table, other kernels are meant to
demonstrate the effect of different Halide update definitions on instruction count and global memory throughput.

**CUDA version**: source code from GPU efficient recursive filtering and summed area table (SIGGRAPH 2011), [Nehab et al.]

Compilation
-----------
- Makefile provided for both projects
- Edit Makefile.common to set the CUDA include path and Halide base path


Profiling files
---------------
The directory <code>nv_profile</code> NVIDIA profiling tools profiling logs. Can be opened using
<code>
$ nvvp cuda_summed_table.nvvp
$ nvvp hl_summed_table.nvvp
</code>

Generated ptx and stamement files
---------------------------------
The directory <code>ptx</code> and <code>stmt</code> contains the generated ptx and statement files for the different
Halide kernels. These can be regenerated by:
<code>
$ HL_JIT_TARGET=cuda-gpu_debug HL_DEBUG_CODEGEN=1 ./hl_summed_table 2> hl_summed_table.ptx
</code>
