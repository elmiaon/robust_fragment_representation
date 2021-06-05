# emd_lib

## Install
Download this directory directly from GitHub, or using command line:
```bash
$ git clone https://github.com/NattapolV/emd_lib.git 
```
Then go into the emd_lib diretory and run command:
```bash
$ pip install .
```

## Example : EMD

```python
# importng emd module
import emd

# initializing weights for each distribution
w1 = [1,1,1,1]
w2 = [1,1,1,1]

# initializing cost_matrix
cost_matrix = [[676,729,784,841],
               [256,289,324,361],
               [36,49,64,81],
               [1296,1369,1444,1521]]

# calling function
e = emd.emd(w1,w2,cost_matrix)

# making an ouput single-precision floating point which equals to emd.c output
e = round(e,6)
```

## Example : K-EMD
```python
# importing emd module
import emd

# defining the number of features
n = 24

# initializing distribution a
d1 = [0.47297,-0.39603,-0.72191,-1.1564,-1.2107,-1.0478,-0.77622,-1.3193,-0.83053,0.25572,0.96179,1.1247,0.5816,0.20141,-0.23309,-0.34172,-0.66759,-0.72191,-0.45034,-0.015841,0.79885,2.2653,1.9394,1.2877]

# initializing distribution b
d2 = [-1.0987,-1.1346,-1.6003,-1.7078,-1.6362,-1.457,-1.0271,-0.70462,0.47771,1.2301,1.2659,1.1226,0.76434,0.65685,0.51354,0.62103,0.72851,0.37023,0.40606,0.37023,0.0836,1.0151,0.62103,0.11943]

# initializing cost matrix from distribution a,b (24x24)
cost_matrix = [[abs(i-j) for j in d2] for i in d1]

# defining infinity
inf = max([sum(row) for row in cost_matrix])

# initializing weights for each distribution
w1 = [1 for i in range(n)]
w2 = [1 for i in range(n)]

# adding more weights to pseudo node
w2 += [n]

# initializing k-cost matrix from cost matrix which dimension is (24x25)
k_cost_matrix = [cost_matrix[i] + [inf] for i in range(n)]

# sorting values in cost_matrix,so that we can remove the maximum value in k-cost matrix easily (decreasing k)
sorted_matrix = [sorted(enumerate(c), key = lambda x : x[1]) for c in cost_matrix]

# iterating over reversed k (from 23 to 0) 
for k in reversed(range(24)):

    # iterating over row in k-cost matrix
    for index,row in enumerate(k_cost_matrix):
    
        # getting maximum value with index
        index_max, val_max = sorted_matrix[index].pop()
        
        # replacing maximum value in k-cost matrix with infinity
        k_cost_matrix[index][index_max] = inf
        
        # replacing value in pseudo node with that maxmimum value
        k_cost_matrix[index][-1] = val_max
    
    #calculating emd and rounding the result to 6 decimal points
    e = round(emd.emd(w1,w2,k_cost_matrix),6)
    
    #printing result
    print("k = {}, emd = {}".format(k,e))
```
