# Reinforcement learning using Matlab Simulink  

"Differential drive mobile robot to approach point using reinfoercement learning"  

For better look check this note, please visit [HackMD.io](https://hackmd.io/@libernormous/ddmr_rl_matlab)  

by:  
[Liber Normous](https://hackmd.io/@libernormous)  

---

# System description  

## Robot dynamic    
The robot dynamic can be seen [here ![](https://i.imgur.com/RmvkGxz.png)
](https://hackmd.io/@libernormous/dynamic_ddmr).  

## Disturbance  
![](https://i.imgur.com/hRHoQbU.png)  

## Reinforcement agent  
![](https://i.imgur.com/0IDKsXH.png)  
  

### State  
1. Position error.  
2. Heading error.  
3. Position error rate.  
4. Heading error rate.  

### Action  
1. left motor voltage (feeded to sigmoid function).  
2. right motor voltage (feeded to sigmoid function).  

## Reward  
1. Reward of getting closer.  
    a. If $e_p<1$, then $R_1 = (1-e_p)20$ 
    b. Else $R_1 = -0.02e_p$
3. Move away from goal penalty.  
    a. If $\frac{\Delta e_p}{\Delta t}<-0.01$, then $R_2 = 2$  
    b. Else $R_2=-\frac{1}{200}\frac{\Delta e_p}{\Delta t}$  
5. Correct heading reward.  
    a. If $|e_h|<10$, then $R_3=2(10-|e_h|)$  
    b. Else $R_3=-e_h\frac{1}{100}$
7. Heading away from goal penalty.
    a. If $\frac{\Delta |e_h|}{\Delta t}<-0.01$, then $R_4 = 2\frac{\Delta |e_h|}{\Delta t}$     
    b. Else $R_4=-\frac{1}{200}\frac{\Delta |e_h|}{\Delta t}$  

## Actor Network  
![](https://i.imgur.com/1otORrs.png)  
![](https://i.imgur.com/y1GjrRa.png)  

## Critic Network  
![](https://i.imgur.com/6M6pIa1.png)
![](https://i.imgur.com/v1lhDqa.png)  

---

# Result  

## Training result  
![](https://i.imgur.com/8g9u06Y.png)  

## Target approaching result. with and without disturbance  
![](https://i.imgur.com/l0Htpqs.png)  

## Target approaching result. with and without disturbance  
![](https://i.imgur.com/CTRkTjp.png)  

## Following figure of 8  
![](https://i.imgur.com/1lnlckE.png)  
![](https://i.imgur.com/MyJIAia.png)  

---

# Tutorial  
1. For testing, open `testing.m` and run.  
2. For training, open `training.m` and run.  
3. To follow path, use `path_generator.m` to follow point array.  

---

