NVCC ?= nvcc
ARCH ?= sm_53
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo -arch=$(ARCH)
TARGET := softmax_cuda
SRC := softmax_cuda.cu


PRECISION ?= fp16
N ?=  100
REPS ?= 10000
MEASURE ?= kernel


.PHONY: all run run-e2e clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) --precision $(PRECISION) --n $(N) --reps $(REPS) --measure $(MEASURE)

run-e2e: $(TARGET) 
	./$(TARGET) --precision  $(PRECISION) --n $(N) --reps $(REPS) --measure end-to-end
clean:
	rm -f $(TARGET)
