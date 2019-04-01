import numpy as np
# import time

# a = np.array([1,2,3])
# b = np.array([1,2,3])

# print(np.dot(a,b))


# a = np.random.rand(1000000)
# b = np.random.rand(1000000)

# tic = time.time()
# c = np.dot(a,b)
# toc = time.time()

# print("Calculation Time in ms = " + str((toc - tic)*1000))

# c = 0
# tic = time.time()
# for i in range(len(a)):
#     c += a[i]*b[i]
# toc = time.time()
# print("Calculation Time in ms = " + str((toc - tic)*1000))

# a1 = np.array([[1,2,3,4],[3,6,2,4],[1,8,3,2]])
# print(a1.sum(axis=0)) #sum vertically
# print(a1.sum(axis=1)) #sum horizontally
# print((a1.sum(axis=0)).reshape(1,4))
# percentage = 100*a1 / (a1.sum(axis=0)).reshape(1,4) # ng uses reshape for making sure of the dimesion 
# print(percentage)
# percentage = 100*a1 / a1.sum(axis=0) #broadcasting in python
# print(percentage)

# a2 = np.random.randn(4) #Random guassian vector
# print(a2)
# print(a2.shape) #dont use this kind of vector(rank 1 array)
# print(a2.T)
# print(np.dot(a2,a2.T))

# a2 = np.random.randn(4,1)
# print(a2)
# print(a2.shape)
# print(np.dot(a2,a2.T))
# print(np.dot(a2.T,a2))


#Optional Assignment
# def sigmoid(x):
#     # x -- A scalar or numpy array of any size
#     return 1 / (1 + np.exp(-x))
# x3 = np.array([1, 2, 3])
# print(sigmoid(x3))

# def sigmoid_derivative(x):
#     s =  1 / (1 + np.exp(-x))
#     ds = s * (1-s)  
#     return ds

# def image2vector(image):
#     """
#     Argument:
#     image -- a numpy array of shape (length, height, depth)

#     Returns:
#     v -- a vector of shape (length*height*depth, 1)
#     """
#     v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),1)
#     return v

# image = np.array([[[ 0.67826139,  0.29380381],
#         [ 0.90714982,  0.52835647],
#         [ 0.4215251 ,  0.45017551]],

#        [[ 0.92814219,  0.96677647],
#         [ 0.85304703,  0.52351845],
#         [ 0.19981397,  0.27417313]],

#        [[ 0.60659855,  0.00533165],
#         [ 0.10820313,  0.49978937],
#         [ 0.34144279,  0.94630077]]])

# print ("image2vector(image) = " + str(image2vector(image)))




def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    Argument:
    x -- A numpy matrix of shape (n, m)    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    #using numpy built-in functions
    #x_norm = np.linalg.norm(x,axis=1,keepdims=True)

    #manually
    x_norm = x**2
    xx = x_norm.sum(axis=1)
    xx = np.sqrt(xx)
    x = x/xx.reshape(2,1)
    return x

x4 = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x4)))




def L1(yhat, y):
    loss = (abs(yhat-y)).sum() 
    return loss

def L2(yhat, y):
    loss = ((yhat - y)**2).sum()
    return loss