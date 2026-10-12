NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= native

agent: agent.cu
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -o agent agent.cu

clean:
	rm -f agent

.PHONY: clean