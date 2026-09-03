#include <iostream>

int main()
{
  constexpr long long number_count = 1'000'000;
  long long sum = 0;

#pragma omp parallel for default(none) reduction(+ : sum)
  for (long long number = 1; number <= number_count; ++number) {
    sum += number;
  }

  const long long expected = number_count * (number_count + 1) / 2;
  std::cout << "Parallel sum: " << sum << '\n';
  std::cout << "Expected sum: " << expected << '\n';

  return sum == expected ? 0 : 1;
}
