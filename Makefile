# Compiler (override on macOS: make CXX=g++-15)
CXX = g++

# Build flags:
#   -O3              speed optimizations
#   -march=native    use this machine's CPU features
#   -flto            link-time optimization
#   -std=c++17       C++17
#   -fopenmp         OpenMP parallelism
#   -IInclude        look for headers in Include/
#   -static-libstdc++  static-link libstdc++ (Linux clusters)
OMPFLAGS = -O3 -march=native -flto -std=c++17 -fopenmp -IInclude -static-libstdc++

# Shared sources (everything under Lib/ except the main Solver.cpp)
COMMON_SRC = Src/Main.cpp \
             Lib/Bucket_Partitioned_MDS/CVRP.cpp \
             Lib/Bucket_Partitioned_MDS/Solution.cpp \
             Lib/Command_Line_Args.cpp \
             Lib/Initializer.cpp \
             $(shell find Lib/Utils -name '*.cpp')

# Main solver (default)
TARGET = Bin/bucket-partitioned-MDS

# Benchmarking / ablation binaries
TARGET_SET = Bin/bucket-partitioned-MDS-set
TARGET_DFS = Bin/bucket-partitioned-MDS-dfs
TARGET_BFS = Bin/bucket-partitioned-MDS-bfs
TARGET_BKT = Bin/bucket-partitioned-MDS-buckets

BENCH_TARGETS = $(TARGET) $(TARGET_SET) $(TARGET_DFS) $(TARGET_BFS) $(TARGET_BKT)

# Default: main solver only
all: $(TARGET)

# Optional CUDA Prim / 2-opt (per-bucket; no full N×N). Needs nvcc + cudart.
NVCC ?= nvcc
CUDA_ARCH ?= -arch=sm_90
CUDA_SRC = Lib/Cuda/cuda_hotpath.cu
TARGET_CUDA = Bin/bucket-partitioned-MDS-cuda

# All variants used for benchmarking
bench-marking: $(BENCH_TARGETS)

$(TARGET): $(COMMON_SRC) Lib/Bucket_Partitioned_MDS/Solver.cpp
	@mkdir -p Bin
	$(CXX) $(OMPFLAGS) $^ -o $@
	@echo "Build successful: $@"

# H100 = sm_90; override CUDA_ARCH for other GPUs.
cuda: $(TARGET_CUDA)

$(TARGET_CUDA): $(COMMON_SRC) Lib/Bucket_Partitioned_MDS/Solver.cpp $(CUDA_SRC)
	@mkdir -p Bin
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH) -Xcompiler "-fopenmp -O3 -std=c++17" \
		-DBPMDS_USE_CUDA -IInclude \
		$(COMMON_SRC) Lib/Bucket_Partitioned_MDS/Solver.cpp $(CUDA_SRC) \
		-o $@ -lgomp -lcudart
	@echo "Build successful: $@ (CUDA Prim/2-opt)"

$(TARGET_SET): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_cpp_set.cpp
	@mkdir -p Bin
	$(CXX) $(OMPFLAGS) $^ -o $@
	@echo "Build successful: $@"

$(TARGET_DFS): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_Non_Lazy_DFS.cpp
	@mkdir -p Bin
	$(CXX) $(OMPFLAGS) $^ -o $@
	@echo "Build successful: $@"

$(TARGET_BFS): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_BFS.cpp
	@mkdir -p Bin
	$(CXX) $(OMPFLAGS) $^ -o $@
	@echo "Build successful: $@"

$(TARGET_BKT): $(COMMON_SRC) Scripts/BenchmarkingCode/Solver_buckets.cpp
	@mkdir -p Bin
	$(CXX) $(OMPFLAGS) $^ -o $@
	@echo "Build successful: $@"

# Remove everything in Bin/
clean:
	rm -rf Bin/*
	@echo "Cleaned Bin/"

.PHONY: all bench-marking clean cuda
