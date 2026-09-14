# Historical SpGEMM phase profiling

The general-kernel phase profiler, source generator and scheduling/lookup
ablations have been removed with the alternate SpGEMM implementations.
Only `OmpHashSpGEMM` in [mtSpGEMM.h](../../include/kernel/mtSpGEMM.h) is maintained.
No phase-level cause should be attributed to it from the old general-kernel
measurements. See the [current source mapping](SourceMapping.md) and
[native timing report](../results/combblas/spgemm/MappedTimingResults.md).

Raw historical phase data remain under
`benchmarks/results/combblas/spgemm/phases-bigred-8166903` and
`benchmarks/results/combblas/spgemm/phases-bigred-8168164`.
The read-only `scripts/analyze_phases.py` can inspect the retained CSV records;
it does not generate or select a multiplication implementation.
