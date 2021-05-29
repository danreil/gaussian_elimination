""" Implement Gaussian elimination
Solve n x n system of equations (Ax=b), with coeffiecents in input matrix A, and with rhs of equation b
Note: obviously numpy or scipy could be used (more efficiently), just for practice
"""

import numpy as np

class ElemOp:
    """ Instance variables are input, left hand side matrix A and right hand side solution vector, b, in the Ax=b equation"""
    def __init__(self, A, b):
        self.A = A
        self.b = b
        dims = A.shape
        self.n_rows, self.n_cols = dims[0], dims[1]  # number of rows and columns of A. Instance attributes
    
    def swap_rows(self, row1, row2):
        # swap positions of row1 and row2 in A and b
        self.A[[row1,row2],:] = self.A[[row2, row1],:]  #use advanced indexing for swapping the rows
        self.b[row1], self.b[row2] = self.b[row2], self.b[row1]
    
    def scalar_mult(self, row, scalar):
        # multiply a row by the given scalar
        if scalar == 0:
            print("Can't multiply by 0")
            return None
        self.A[row] = scalar*self.A[row]
        self.b[row] = scalar*self.b[row]

    def add_to_row(self, row1, row2, scalar):
        # add a scalar multiple of row 1 to row 2 
        self.A[row2] = scalar*self.A[row1] + self.A[row2]
        self.b[row2] = scalar*self.b[row1] + self.b[row2]    
        
    def is_row_zeros(self, row):
        # return true if every entry of A[row] is zero, so that row can be moved to bottom
        #print(f'checking row {row}')
        if np.all(self.A[row] == 0):
            print(f'row {row} is all zeros')
            return True
        return False
    
    def move_zero_rows(self):
        # if a row is all zeros, move that row to the bottom of the matrix to get in row echelon order. Use swap rows method.
        n = self.n_rows-1  #n initially is number of rows (minus 1 since 0-indexed). After swapping rows, decrement it 
        i = 0
        while i < n+1:
            if self.is_row_zeros(i):
                self.swap_rows(i,n)
                i +=1
                n -=1
            else:
                i +=1

    def clear_pivot_column(self, pivot):
        """Make entries below a pivot element in column k all zero. Pivot element lies on diagonal, i.e. A[k,k] for k=0,1...n-1.
        If pivot element at A[k,k]=0, swap that row with bottom-most non-zero row"""
        for k in range(pivot+1,self.n_rows):
            factor = -self.A[k,pivot]/self.A[pivot,pivot]
            self.add_to_row(pivot, k, factor)
    #TODO: add the functionality for detecting zero and moving to bottom
    
    def iter_pivots(self):
        """Iterate over the pivots, clearing each column below each pivot"""
        for k in range(self.n_cols-2):  #n_cols-1 would the last column, but this column does not need to be cleared
            self.clear_pivot_column(k)         
    
def back_sub(A, b, n_rows, n_cols):
    """Input coefficient matrix A, and rhs vector b, with n_rows and n_cols input. All of these are instance variables of an object of type ElemOp.
    In Ax=b, let x be a vector, [x(0), x(1)...x(n)], (where n=n_rows-1) that solves the equation. After applying the operations from ElemOp, should have an upper
    triangular matrix. So, solving, x(n)=b(n)/A(n,n) and in general, x(i)=[b(i) - SUM[A(i,j)x(j), from j=i+1 to n]]/A(i,i).
    """
    x_vec = np.zeros((n_rows,1))  #create zero vector with correct dimensions.
    n = n_rows-1  #Just for easier indexing
    x_vec[n] = b[n]/A[n,n]
    for i in reversed(range(n)):  #iterate backwards over the rows, from n to 0 inclusive
        out = back_sub_sum(A[i], x_vec, i, n_cols)
        x_vec[i] = (b[i] - out)/A[i,i]
    return x_vec
      
def back_sub_sum(A_vec, x_vec, row_num, n_cols):
    """Helper function for back_sub function. Input the row vector of A being processed by the loop in back_sub,
    and the x_vec solution, which is a vector of zeros with one entry solved for in each iteration of the loop.
    Mainly using this to get the indexing working in back_sub"""
    out = 0
    for j in range(row_num+1,n_cols):
        out += A_vec[j]*x_vec[j]
    return out
                       
A=np.array([[1,0,0],[0,1,0],[0,1,1]], dtype=float)
b=np.array([4,5,6], dtype=float)
x=ElemOp(A,b)
print(x.A, x.b)
#x.move_zero_rows()
x.iter_pivots()
print(x.A, x.b)
solution = back_sub(x.A, x.b, x.n_rows, x.n_cols)
print(solution)






