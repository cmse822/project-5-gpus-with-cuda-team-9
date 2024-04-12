# Project 5: GPU Computing with CUDA

## Warm-up

In this project,  you will write 2 CUDA kernels for doing heat diffusion in one
dimension. The first kernel will be a naive implementation, the second kernel
will leverage CUDA shared memory. Be sure to watch and read the material in the [associated pre-class assignments](../schedule.md)! Then, as you are developing and running your CUDA code, refer to the following ICER documentation pages for using GPUs on HPCC:

- [Compiling for GPUs](https://docs.icer.msu.edu/Compiling_for_GPUs/)
- [Requesting GPUs](https://docs.icer.msu.edu/Requesting_GPUs/)

I strongly recommend using HPCC for this project.

## Part 1 and Part 2

The final version of the `diffusion.cu` is in the repo.

## Part 3

We can see that the diffusion over 1e6 timesteps progresses as we owuld expect:
![P2-comparison-Latency](./fig2.png)

Also we see that the errors between host and cuda are within acceptable limits
![P2-comparison-Latency](./fig1.png)

Running for 1e6 timesteps with a domain size of 2^11 we observe these timings with the -O3 flag:\
Host function took: 0.00333578ms per step \
Cuda Kernel took: 0.00211785ms per step\
Shared Memory Kernel took: 0.00213615ms per step\
Excessive cudaMemcpy took: 0.0196426ms per step\

And these timings without the -O3 flag: \
Host function took: 0.0150383ms per step\
Cuda Kernel took: 0.00201429ms per step\
Shared Memory Kernel took: 0.00203554ms per step\
Excessive cudaMemcpy took: 0.0170543ms per step \

These timings are in line with what we expect since shared vs global cuda memory doesn't matter very much at these scales. An interesting finding however is just how much faster the -O3 flag makes the CPU diffusion. On the other hand just as we expected the excessive memory copy cuda code is just as slow as the unoptimized CPU diffusion.

Now testing with a domain size of 2^15 and the optimization we see these timings:\
Host function took: 0.05409ms per step\
Cuda Kernel took: 0.0035072ms per step\
Shared Memory Kernel took: 0.0029344ms per step\
Excessive cudaMemcpy took: 0.0902954ms per step\

And without optimizing:\
Host function took: 0.241148ms per step\
Cuda Kernel took: 0.00358432ms per step\
Shared Memory Kernel took: 0.00294176ms per step\
Excessive cudaMemcpy took: 0.0919658ms per step\

Again we see the same trends as before except that we are approaching the scales where the shared memory is starting to become more efficient than the regular CUDA implementation.

with a block dimension of 256 we observe:\
Host function took: 0.05409ms per step\
Cuda Kernel took: 0.0035072ms per step\
Shared Memory Kernel took: 0.0029344ms per step\
Excessive cudaMemcpy took: 0.0902954ms per step\

Now testing with a block dimension of 512 we observe:\
Host function took: 0.048089ms per step\
Cuda Kernel took: 0.00365056ms per step\
Shared Memory Kernel took: 0.00369536ms per step\
Excessive cudaMemcpy took: 0.091873ms per step\

And with a block size of 1024:\
Host function took: 0.0558209ms per step\
Cuda Kernel took: 0.00365024ms per step\
Shared Memory Kernel took: 0.00285376ms per step\
Excessive cudaMemcpy took: 0.0955904ms per step\


![P2-comparison-Latency](./Speedups.png)\
Here we can observe the slight slowdowns when we go to higher block sizes. Furthermore the CPU with optimizations is better than the excess memory CUDA and if we parallized the CPU code we would expect to be even faster than the best CUDA implementation for this problem.






## What to turn In

Your code, well commented, and answers to these questions:

1. Report your timings for the host, naive CUDA kernel, shared memory CUDA kernel,
and the excessive memory copying case, using block dimensions of 256, 512,
and 1024. Use a grid size of `2^15+2*NG` (or larger) and run for 100 steps (or
shorter, if it's taking too long). Remember to use `-O3`!

`Host function took: 0.05409ms per step\
Cuda Kernel took: 0.0035072ms per step\
Shared Memory Kernel took: 0.0029344ms per step\
Excessive cudaMemcpy took: 0.0902954ms per step\`

2. How do the GPU implementations compare to the single threaded host code. Is it
faster than the theoretical performance of the host if we used all the cores on
the CPU?

  Now testing with a domain size of 2^15 and the optimization (all cores) we see these timings:\
  `Host function took: 0.05409ms per step\
  Cuda Kernel took: 0.0035072ms per step\`
  
  And without optimizing:\
  `Host function took: 0.241148ms per step\
  Cuda Kernel took: 0.00358432ms per step\`

  GPU is outpreform than the single threaded host code and is also better than the theoretical preformance of the host with all cores.


3. For the naive kernel, the shared memory kernel, and the excessive `memcpy` case,
which is the slowest? Why? How might you design a larger code to avoid this slow down?

  Based on the  results, the slowest operation is the excessive cudaMemcpy case, which took 0.091873ms per step. This slowdown is due to the overhead incurred by repeatedly copying data between the host and device memory. Each cudaMemcpy operation involves transferring data over the PCI Express bus, which introduces latency and can significantly impact performance, especially if done excessively.
  
  To avoid this slowdown, we can employ strategies to minimize unnecessary data transfers between the host and device:
  
  * Data Organization: Organize your data in such a way that minimizes the need for frequent transfers between host and device memory. Try to keep data that will be frequently accessed by the GPU in device memory for as long as possible.

  * Shared Memory Usage: Leverage shared memory within CUDA kernels to reduce the need for accessing global memory. Shared memory accesses are typically faster than global memory accesses due to their lower latency and higher bandwidth. Optimize your kernel to utilize shared memory efficiently for storing frequently accessed data.

4. Do you see a slow down when you increase the block dimension? Why? Consider
that multiple blocks may run on a single multiprocessor simultaneously, sharing
the same shared memory.

  There appears to be a slight slowdown when increasing the block dimension from 256 to 512 and then to 1024. When we increase the block dimension, each block requires more resources, such as registers and shared memory, which could potentially lead to fewer blocks being able to run concurrently on a single multiprocessor (SM).


