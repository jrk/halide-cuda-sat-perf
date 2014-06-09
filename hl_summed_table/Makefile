PROGRAM_NAME=hl_summed_table

include ../Makefile.common

BUILD_DIR=_build
BIN_DIR=../bin
EXE=$(BIN_DIR)/$(PROGRAM_NAME)

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC := $(wildcard *.cpp)
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

all: $(EXE)

$(BUILD_DIR)/%.o: %.cpp $(INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -c $< -o $@ -fno-rtti -I$(HALIDE_DIR)/include/

$(EXE): $(OBJ) $(HALIDE_LIB)
	$(MKDIR) $(BIN_DIR)
	$(CXX) -lz -lpthread -ldl -g -Wall $(OBJ) $(HALIDE_LIB) -o $(EXE)

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(EXE)
