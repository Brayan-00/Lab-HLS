CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native
LDFLAGS = -lm

TARGET = softmax_cpu
SRC = softmax_cpu.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
