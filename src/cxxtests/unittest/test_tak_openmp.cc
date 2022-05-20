#include "../doctest.h"

#include <stdio.h>
#include <omp.h>

TEST_CASE("openmp")
{

  int partial_Sum(0), total_Sum(0);
  #pragma omp parallel firstprivate(partial_Sum) shared(total_Sum)
  {
    printf("total thread %d, thread ID %d\n", omp_get_num_threads(),  omp_get_thread_num());
    partial_Sum = 0;
    total_Sum = 0;

    #pragma omp for
    for(int i = 1; i <= 1000; i++){
      partial_Sum += i;
    }


    //Create thread safe region.
    #pragma omp critical
    {
        //add each threads partial sum to the total sum
        total_Sum += partial_Sum;
    }
  }
  printf("Total Sum: %d\n", total_Sum);
  CHECK(total_Sum==500500);
}
