# SPCraft examples

The examples are grouped by programming model:

- `serial/`: sequential C++ examples, including direct Matrix Market input with
  `fast_matrix_market`.
- `openmp/`: shared-memory examples using OpenMP threads.
- `mpi/`: distributed-memory examples using MPI processes.
- `python/`: Python and SciPy examples.
- `metaprogramming/`: modern C++ template and policy examples.

After configuring and building SPCraft, run the MPI introduction with:

```bash
mpirun -n 4 ./build/examples/mpi/e04_mpi_hello
```

The MPI example is built only when CMake finds a C++ MPI implementation.
