NVCC ?= nvcc
ARCH ?= sm_53
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo -arch=$(ARCH)

TARGET := softmax_cuda
SRC := softmax_cuda.cu

PRECISION ?= fp16
INPUT ?= inputs/vector_N10.csv
REPS ?= 10000
MEASURE ?= kernel
MEASUREMENTS ?= 5

.PHONY: all run run-e2e clean

all: $(TARGET)

$(TARGET): $(SRC)
        $(NVCC) $(NVCCFLAGS) -o $@ $<

# Medición del kernel
run: $(TARGET)
        ./$(TARGET) --precision $(PRECISION) --input $(INPUT) --reps $(REPS) --measure kernel --measurements $(MEASUREMENTS)

# Medición end-to-end
run-e2e: $(TARGET)
        ./$(TARGET) --precision $(PRECISION) --input $(INPUT) --reps $(REPS) --measure end-to-end --measurements $(MEASUREMENTS)

clean:
        rm -f $(TARGET)
