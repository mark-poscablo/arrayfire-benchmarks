#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <vector>

using af::array;
using af::deviceGC;
using af::dim4;
using af::pinverse;
using af::randu;
using std::vector;

static void pinverseBench(benchmark::State& state, af_dtype dtype) {
    dim_t dim0 = state.range(0);
    dim_t dim1 = state.range(1);
    dim_t dim2 = state.range(2);
    dim_t dim3 = state.range(3);

    array in = randu(dim0, dim1, dim2, dim3, dtype);
    for (auto _ : state) {
        array out = pinverse(in);
        out.eval();
    }

    deviceGC();
}

int main(int argc, char** argv) {
    vector<af_dtype> dtypes = {f32, f64, c32, c64};

    af::benchmark::RegisterBenchmark("pinverseDim0GtDim1", dtypes, pinverseBench)
        ->RangeMultiplier(2)
        ->Ranges({{1<<3, 1<<9}, {1<<2, 1<<8}, {1, 1}, {1, 1}})
        ->ArgNames({"dim0", "dim1", "dim2", "dim3"})
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::RegisterBenchmark("pinverseDim1GtDim0", dtypes, pinverseBench)
        ->RangeMultiplier(2)
        ->Ranges({{1<<2, 1<<8}, {1<<3, 1<<9}, {1, 1}, {1, 1}})
        ->ArgNames({"dim0", "dim1", "dim2", "dim3"})
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::RegisterBenchmark("pinverseBatch3D", dtypes, pinverseBench)
        ->RangeMultiplier(1)
        ->Ranges({{1<<6, 1<<6}, {1<<5, 1<<5}, {1, 6}, {1, 1}})
        ->ArgNames({"dim0", "dim1", "dim2", "dim3"})
        ->Unit(benchmark::kMicrosecond);

    benchmark::Initialize(&argc, argv);
    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
