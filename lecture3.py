import time
import numpy as np

# mylist = np.random.randint(0, 10, (3,3))
# sub_array = mylist[:2, :2]
# row_col = mylist[0, :]
# print(mylist, '\n', sub_array)

mylist = np.arange(1, 10)
# shape the array into a matrix by using the method 'reshape.()'
column = mylist.reshape((3, 3))
# col = mylist.reshape((1,9))
col = mylist[np.newaxis, :]  # this dim here bf the comma must be a new dim
# column = mylist[:,:,np.newaxis]
# print(mylist,'\n ', '-----------------------','\n',  grid, '\n', '---------------------', '\n', column)

array1 = [1, 2, 3]
array2 = [3, 2, 1]
new = np.concatenate([array1, array2])
# print(new)

# stacking cols
array3 = np.array([[1, 2, 3], [3, 2, 1]])
x = np.array([[7, 8, 9], [3, 5, 4]])
new2 = np.hstack([array3, x])
# print(new2)

array4 = np.arange(16).reshape((4, 4))
upper, lower = np.hsplit(array4, [2])
x1, x2, x3 = np.split(array4, [3, 5])  # list of indexes to split
# print(upper, lower)


#
np.random.seed(0)


def compute_reciprocals(values):
    output = np.empty(len(values))  # free allocating in memory m array ,
    # i have to compute the reciprocal for each of the value given in input
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output


values = np.random.randint(1, 100, size=1000)
start_time = time.time()
out = compute_reciprocals(values)
end_time = time.time() - start_time
# print(end_time)

# ufuncs:
start_time = time.time()
# out = 1.0/values
out = 1.0 + values
end = time.time() - start_time
np.add(1, values)
print(out)
