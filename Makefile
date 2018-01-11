# Set PQR_DIR to build pqrRand into your executable (using the %.x rule)
PJ_DIR = ./lib
#~ PQR_INC = $(PQR_DIR)/include

LOCAL_DIR = $(HOME)/local

CXX = gcc
MARCH = native
STD = c++11 
# GCC flags, including many useful warnings
STABILITY_FLAGS = -pedantic-errors -fno-common -mfpmath=sse -mieee-fp #sse flag to avoid weird x87 registers (see https://gcc.gnu.org/wiki/FloatingPointMath)
STABILITY_WARNINGS = -Wall -Wextra -W -Wconversion -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Wmissing-field-initializers -Wlogical-op -Wunsafe-loop-optimizations -Wwrite-strings -Wundef -Wfloat-equal
PERFORMANCE_FLAGS = -O2 -march=$(MARCH) -msse4 -mavx2 -Winline -Wdisabled-optimization -Wpadded -ftree-vectorize # vectorize is the only thing from O3 that we want
BUILD_LIB_FLAGS = -fPIC
# 
CXXFLAGS = -std=$(STD) $(STABILITY_WARNINGS) $(PERFORMANCE_FLAGS) $(BUILD_LIB_FLAGS)

# The directory structure of pqRand
INCLUDE = ./include
SOURCE = ./source
EXAMPLES = ./examples

INC_FLAGS = -I $(INCLUDE) -I $(LOCAL_DIR)/include
LOCAL_LIBS = -lpqr -lzLib -lpythia8
LIB_FLAGS = -lstdc++ -lm -lQtCore -lfastjet -L $(LOCAL_DIR)/lib $(LOCAL_LIBS)  -L $(PJ_DIR) -lpj

EXAMPLES_CPP = $(wildcard examples/*.cpp)
EXAMPLES_X = $(patsubst %.cpp, %.x, $(EXAMPLES_CPP))

FILENAMES = ArrogantDetector SpectralPower 
OBJS = $(addsuffix .o, $(addprefix $(SOURCE)/, $(FILENAMES)))

all : libpqr.so $(EXAMPLES_X)

libpj.so: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) $(LIB_FLAGS) -o $(PJ_DIR)/$@

%.x : %.cpp # $(PQR_DIR)/libpqr.so
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) $(LIB_FLAGS) $*.cpp -o $@
	
%.o : %.cpp 
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) $(LIB_FLAGS) $*.cpp -c -o $*.o
	
.PHONY: clean

clean:
	rm -f $(SOURCE)/*.o
#	rm -f $(EXAMPLES_X)
	rm -f $(PJ_DIR)/libpj.so
