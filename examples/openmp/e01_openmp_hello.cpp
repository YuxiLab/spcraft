#include <omp.h>

#include <iostream>

int main()
{
  std::cout << "OpenMP will use " << omp_get_max_threads() << " threads.\n";

#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();

#pragma omp critical
    std::cout << "Hello from thread " << thread << " of " << thread_count << ".\n";
  }

  return 0;
}
