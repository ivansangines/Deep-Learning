import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    #d = 2*5.2 + 4 * 6.7 + 6 * 9.1 + 8 * 10.9
    #print(d)
    #e = 2*5.2 + 2*6.7 +2*9.1 + 2*10.9
    #print(e)
    x = np.ndarray((6,1))
    #x = np.zeros((6,1))
    #x = x.astype(float)
    y = np.ndarray((6,1))
    #y = np.zeros((6,1))
    #y = y.astype(float)
    x[0,0] = 1
    x[1,0] = 2
    x[2,0] = 3
    x[3,0] = 4
    x[4,0] = 5
    x[5,0] = 6
    y[0,0] = 3.2
    y[1,0] = 6.4
    y[2,0] = 10.5
    y[3,0] = 17.7
    y[4,0] = 28.1
    y[5,0] = 38.5
    a=b=c=d=0
    a1=b1=c1=d1=0
    a2=b2=c2=d2=0
    for i in range(len(x)):
        a += x[i,0] * x[i,0] * x[i,0] * x[i,0]
        a1 += x[i,0] * x[i,0] * x[i,0]
        a2 += x[i,0] * x[i,0]

        b += x[i,0] * x[i,0] * x[i,0]
        b1 += x[i,0] * x[i,0]
        b2 += x[i,0]

        c += x[i,0] * x[i,0]
        c1 += x[i,0]
        c2 += 1

        d += y[i,0] * x[i,0] * x[i,0]
        d1 += y[i,0] * x[i,0]
        d2 += y[i,0]

    A = np.ndarray((3,3))
    #A = np.zeros((3,3))
    #A = A.astype(float)
    A[0,0] = a  #int(a) #2275
    A[0,1] = b #int(b) #441
    A[0,2] = c #int(c) #91
    A[1,0] = a1 #int(a1) #379.16666667
    A[1,1] = b1 #int(b1) #73.5
    A[1,2] = c1 #int(c1) #15.1666667
    A[2,0] = a2 #int(a2) #63.19444
    A[2,1] = b2 #int(b2) #12.25
    A[2,2] = c2 #int(c2) #2.527778
    ainv = np.linalg.inv(A) # Doing inverse of A
    z = np.ndarray((3,1)) #Results
    #z = np.zeros((3,1))
    #z = z.astype(float)
    z[0,0] = d
    z[1,0] = d1
    z[2,0] = d2
    res = np.dot(ainv,z) # a = res[0,0] and b=[1,0]
    print(res)
    # do a scatter plot of the data
    area = 10
    colors = ['black']
    plt.scatter(x, y, s=area, c=colors, alpha=0.5, linewidths=8) #drawing points using X,Y data arrays
    plt.title('Linear Least Squares Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    yfitted = x * x * res[0,0] + x * res[1,0] + res[2,0]
    line, = plt.plot(x, yfitted, '--', linewidth=2) #line plot
    line.set_color('red')
    plt.show()
    


if __name__ == "__main__":
	sys. exit(int(main() or 0))