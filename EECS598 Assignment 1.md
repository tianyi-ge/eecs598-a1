# EECS598 Assignment 1

## Task1

### 1. Architecture choice

### 2. Profiling

Here we use

```sh
nohup psrecord "python p0.py" --log q2.log --interval 0.5 --include-children --plot plot.png &
```

to profile the training process. The sampling interval is 0.5 s (the choice of sampling intervals will influence the total completion time).

1. Completion time of training: **<u>~7952.41</u>** seconds
2. Average CPU usage: **<u>542.10%</u>**. Periodically, spikes occur at the end of each batch because (...).
3. Average memory usage: **<u>2742.22</u>** MB. Periodically, spikes occur at the end of each batch because (...).

<img src="/home/tianyi/Documents/21 Winter/EECS598/Assignment/eecs598-a1/q2.png" style="zoom:67%;" />

### 3. With chkpt

```sh
nohup psrecord "python p0.py --do_chkpt" --log q3.log --interval 0.5 --include-children --plot q3.png &
```

The completion time (with `psrecord` enabled for fair comparison with Question 2) with checkpointing is **<u>~7954.38</u>** seconds. It's only 2 seconds more than the training process without checkpointing. The only metric that increases significantly is memory usage, which on average is **<u>4430.49</u>** MB.

### 4. Benchmark

#### 4.1 Batch_size = 32

```sh
nohup psrecord "python p0.py --batch_size=32" --log q4.1.log --interval 0.5 --include-children --plot q4.1.png &
```

1. Completion time of training: **<u>~11089.77</u>** seconds.
2. Average CPU usage: **<u>621.53%</u>**. 
3. Average memory usage: **<u>4410.67</u>** MB.
4. Compared to Question 2 (batch size = 64),
   1. The completion time increases by **<u>~39%</u>**.
   2. The CPU usage increases by **<u>~15%</u>**.
   3. The memory usage increases by **<u>~1.6%</u>**.

<img src="/home/tianyi/Documents/21 Winter/EECS598/Assignment/eecs598-a1/q4.1.png" alt="q4.1" style="zoom:67%;" />

#### 4.2 Batch_size = 256

```sh
nohup psrecord "python p0.py --batch_size=256" --log q4.2.log --interval 0.5 --include-children --plot q4.2.png &
```

1. Completion time of training: **<u>~6111.02</u>** seconds

2. Average CPU usage: **<u>389.76%</u>**.

3. Average memory usage: **<u>4459.34</u>** MB.

4. Compared to Questions 2 (batch size = 64), 

   1. The completion time drops by **<u>~23%</u>**.
   2. The CPU usage drops by **<u>~28%</u>**.
   3. The memory usage increases by **<u>~63%</u>**.

   <img src="/home/tianyi/Documents/21 Winter/EECS598/Assignment/eecs598-a1/q4.2.png" alt="q4.2" style="zoom:67%;" />

#### 4.3. Comparisons

In terms of the network convergence, the accuracy convergence is **<u>slower</u>** (see section 5) as the batch size increases because although the gradient calculation is closer to the global result, the larger batch size might lead to a saddle point or local minimum since the loss function is usually not smooth.

### 5. Tensorboard

<img src="/home/tianyi/.config/Typora/typora-user-images/image-20210303193904075.png" alt="image-20210303193904075" style="zoom:67%;" />

## Task 2

### 6. Distributed training

To start the i-th worker on the i-th node in a 3 node cluster,

```sh
i=0 nohup psrecord "python p0.py -n 3 -np 1 -nr $i" --log q6.$i.log --interval 0.5 --include-children --plot plot.png &
```

#### 6.1 One worker

#### 6.2 Two workers

#### 6.3 Three workers

### 7. Two processes per worker

```sh
nohup psrecord "python p0.py -np 2" --log q7.log --interval 0.5 --include-children --plot q7.png &
```

1. Completion time of training: **<u>~6036.68</u>** seconds
2. Average CPU usage: **<u>1083.77%</u>**.
3. Average memory usage: **<u>5354.05</u>** MB.
4. Compared to Questions 2 (1 processor per worker), 
   1. The completion time drops by **<u>~24%</u>**.
   2. The CPU usage increases by **<u>~100%</u>**.
   3. The memory usage increases by **<u>~95%</u>**.

<img src="/home/tianyi/Documents/21 Winter/EECS598/Assignment/eecs598-a1/q7.png" alt="q7" style="zoom: 67%;" />

Starting 2 processes on one single worker, from the observation of CPU and memory, basically doubles the CPU and memory usage. As the completion time, it saves **<u>~24%</u>** of the execution time. However, the accuracy drops **<u>~0.05%</u>** if starting two processes.

<img src="/home/tianyi/.config/Typora/typora-user-images/image-20210303194802682.png" alt="image-20210303194802682" style="zoom:67%;" />

### 8. Convergence rate



### 9. Hours



