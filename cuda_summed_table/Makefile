PROGRAM_NAME=cuda_summed_table

include ../Makefile.common

BIN_DIR := ../bin
EXE     := $(BIN_DIR)/$(PROGRAM_NAME)

INCLUDE_FLAGS := -I$(CUDA_INC) -Iinclude/

CUDA_NVCC_FLAGS=-gencode arch=compute_20,code=sm_21 --ptxas-options=-v -ftz=true -prec-div=false -prec-sqrt=false

INC := $(wildcard include/*.h)
OBJ := src/kernel_perf.o src/defs.o src/sat.o src/gpudefs.o

all: $(EXE)

src/sat.o: src/sat.cu $(INC)
	$(NVCC) $(CUDA_NVCC_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

src/gpudefs.o: src/gpudefs.cu $(INC)
	$(NVCC) $(CUDA_NVCC_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

src/defs.o: src/defs.cc $(INC)
	$(CXX) $(INCLUDE_FLAGS) -c $< -o $@

src/kernel_perf.o: src/kernel_perf.cc $(INC)
	$(CXX) $(INCLUDE_FLAGS) -c $< -o $@

$(EXE): $(OBJ)
	$(MKDIR) $(BIN_DIR)
	$(NVCC) $(CUDA_NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	$(RM) $(OBJ) $(EXE)
