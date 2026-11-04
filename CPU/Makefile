# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native
LDFLAGS = -lm

# Target executable and source file
TARGET = softmax_cpu
SRC = softmax_cpu.cpp

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean rule to remove the compiled binary
clean:
	rm -f $(TARGET)
.PHONY: all clean
