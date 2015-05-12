# Median-Filter-CUDA
Median Filter using CUDA in C++

## Build ##
To build the program, just run

```
make [optional: use -j <Number of Processors>]
```

## Running the program ##
```
bin/mf <Filter Size> data/lena.png path/to/output/file
```
Note that the filter size must be 3, 7, 11, or 15.

## Timing ##
Simply run the time Shell script to get sys, user, and real timings for various grid and block configuration we have chosen.
```
./time.sh
```

