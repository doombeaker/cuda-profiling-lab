# Exercise 09: Synchronization Patterns

## Learning Goal

Compare three GPU synchronization methods and see their effect on the nsys timeline. Learn when to use DeviceSynchronize, StreamSynchronize, or EventSynchronize.

## Prerequisite

Exercise 03 (multi-stream), basic profiling

## Key Concepts

- **cudaDeviceSynchronize()**: blocks the CPU until ALL work on ALL streams is done. Coarse-grained, simplest to use, worst for concurrency.
- **cudaStreamSynchronize(stream)**: blocks until the specific stream finishes. Other streams can continue. Medium granularity.
- **cudaEventSynchronize(event)**: blocks until a specific event (a point in a specific stream) is reached. Finest granularity — can signal completion of one kernel without waiting for the whole stream.

## Build & Run

```bash
make && ./profile.sh
```

## nsys Flags Explained

`--trace=cuda` shows all CUDA API calls (sync calls are clearly visible)

## How to Read Results

### Command-line Output

DeviceSynchronize time ≈ 2× one kernel time (serialized). StreamSynchronize and EventSynchronize times ≈ 1× kernel time (overlapped).

### nsys GUI

1. Open `nsys-ui ./report09.nsys-rep`
2. Look at the CPU CUDA API row:
   - `cudaDeviceSynchronize` appears as a long blocking API call
   - `cudaStreamSynchronize` appears as shorter API calls, one per stream
   - `cudaEventSynchronize` appears as very short API calls
3. Look at the GPU timeline:
   - With DeviceSynchronize: the two kernels are serialized (no overlap)
   - With StreamSynchronize/Event: the two kernels overlap (run concurrently)

## Experiment

Add a third stream and test with 3 concurrent kernels vs serialized via DeviceSynchronize.