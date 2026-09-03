#include <omp.h>

#include <iostream>
#include <vector>

int main()
{
  constexpr int element_count = 16;
  std::vector<int> squares(element_count);
  std::vector<int> workers(element_count);

#pragma omp parallel for default(none) shared(squares, workers)
  for (int index = 0; index < element_count; ++index) {
    squares[index] = index * index;
    workers[index] = omp_get_thread_num();
  }

  for (int index = 0; index < element_count; ++index) {
    std::cout << "iteration " << index << " ran on thread " << workers[index]
              << ": square = " << squares[index] << '\n';
  }

  return 0;
}
