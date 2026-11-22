# Rank Predictor

A tool for calculating expected final rank in Japanese mahjong.

## Build

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=</path/to/install>
make
make install
```

## Sample Program

```
$ cd build
$ ./main2
0.2162 0.2789 0.2862 0.2272 
0.4542 0.2792 0.1736 0.0930 
0.1028 0.1690 0.2684 0.4598 
0.2268 0.2729 0.2718 0.2285 
$ ./main3
0.2294 0.2768 0.2712 0.2226 
0.4635 0.2674 0.1724 0.0967 
0.1002 0.1779 0.2699 0.4520 
0.2070 0.2779 0.2864 0.2287 
```
