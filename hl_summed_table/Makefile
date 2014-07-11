PROGRAM_NAME=hl_summed_table

include ../Makefile.common

BUILD_DIR=_build
BIN_DIR=../bin/$(BUILD_PREFIX)
EXE=$(BIN_DIR)/$(PROGRAM_NAME)

HALIDE_LIB=$(HALIDE_DIR)/bin/$(BUILD_PREFIX)/libHalide.a

SRC := $(wildcard *.cpp)
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

all: $(EXE)

$(BUILD_DIR)/%.o: %.cpp $(INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -c $< -o $@ -fno-rtti -I$(HALIDE_DIR)/include/

$(EXE): $(OBJ) $(HALIDE_LIB)
	$(MKDIR) $(BIN_DIR)
	$(CXX) -ldl -lpthread -g -Wall $(OBJ) $(HALIDE_LIB) -lz -lpthread -ldl -o $(EXE)

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(EXE)
