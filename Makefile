CXX      := nvcc
CXXFLAGS := -O3 -std=c++14 -lnvidia-ml -Xcompiler -pthread
TARGET   := gpu_pcie_validator
SRC      := gpu_pcie_validator.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
