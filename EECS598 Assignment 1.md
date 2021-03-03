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
3. Average memory usage: **<u>2742.22</u>**MB. Periodically, spikes occur at the end of each batch because (...).

![](/home/tianyi/Documents/21 Winter/EECS598/Assignment/q2.png)

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

1. Completion time of training: (...)
2. Average CPU usage: **<u>389.76%</u>**. 
3. Average memory usage: 4459.34</u>** MB.
4. Compared to Question 2 (batch size = 64), (...).

#### 4.2 Batch_size = 256

```sh
nohup psrecord "python p0.py --batch_size=256" --log q4.2.log --interval 0.5 --include-children --plot q4.2.png &
```

1. Completion time of training: **<u>~6111.02</u>** seconds

2. Average CPU usage: **<u>389.76%</u>**.

3. Average memory usage: **<u>4459.34</u>** MB.

4. Compared to Questions 2 (batch size = 64), 

   1. The completion time also drops by ~23%.
   2. The CPU usage drops by ~28%.
   3. The memory usage increases by ~63%.

   ![q4.2](/home/tianyi/Documents/21 Winter/EECS598/Assignment/q4.2.png)

   In terms of the network convergence, the accuracy convergence is **<u>slower</u>** than batch size of 64 because as the batch size increases, although the gradient calculation is closer to the global result, the loss function might hit a saddle point or local minimum since the loss function is usually not smooth.

   ![image-20210303142941905](/home/tianyi/.config/Typora/typora-user-images/image-20210303142941905.png)

### 5. TensorBoard

![image-20210303141616428](/home/tianyi/.config/Typora/typora-user-images/image-20210303141616428.png)

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



### 8. Convergence rate

### 9. Hours



