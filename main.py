import torch
from torch.autograd import grad
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from torchviz import make_dot

def cases_1():
    # Both x and y are vectors
    x = torch.tensor([1., 2.], requires_grad=True)

    y = torch.empty(3)
    y[0] = x[0] ** 2
    y[1] = x[0] ** 2 + 5 * x[1] ** 2
    y[2] = 3 * x[1]

    v = torch.tensor([1.0, 2.0, 3.0])
    y.backward(v)  # VJP.

    print("y:", y)
    print("x.grad:", x.grad)

    # z = y[0] + 2 * y[1] + 3 * y[2] # These are equivalent to the above cases
    # z.backward() # Backward is implicit used for dim = 1
    # print('z:', z)
    # print('x.grad', x.grad)

    # Manual computation.
    # [
    #   [2, 0],
    #   [2, 20],
    #   [0, 3]
    # ]
    dydx = torch.tensor(
        [
            [2 * x[0], 0],
            [2 * x[0], 10 * x[1]],
            [0, 3],
        ]
    )

    assert torch.equal(x.grad, v @ dydx)

def cases_2():
    #Both x and y are matrix (compatible in size)
    x = torch.tensor([[1., 2.], [2., 3.], [1., 3.]], requires_grad=True) # 3 x 2

    # 1st dimension of y is not necessarily same as 1st dim of x
    y = torch.empty((3,3)) # 3 x 3
    y[0,0] = x[0,0] ** 2
    y[0,1] = x[1,0] + x[1,1]
    y[0,2] = x[1,0]**2 + 2 * x[2,1]
    y[1,0] = 0
    y[1,1] = x[2,0] ** 2
    y[1,2] = 3 * x[1,1] + 2
    y[2,0] = 0
    y[2,1] = x[0,1]
    y[2,2] = x[0,1]**2

    v = torch.tensor(
        [
            [1., 2., 3.],
            [2., 3., 1.],
            [1., 3., 2.]
        ]
    ) # v has to have the same dimension as x

    y.backward(v)
    print('x.grad using v: ', x.grad)

    # This is equivalent to pick v as above
    # z = y[0,0] + 2 * y[0,1] + 3 * y[0,2] + 2 * y[1,0] + 3 * y[1,1] + y[1,2] + y[2,0] + 3 * y[2,1] + 2 * y[2,2]
    # z.backward()
    # print('x.grad using z: ', x.grad)

    # Manual computation: J = R^{9x6}
    dydx = torch.tensor(
        [
            [2 * x[0,0], 0, 0, 0, 0, 0],# dy [0,0] with respect to all the elements of x
            [0, 0, 1, 1, 0, 0], # dy [0,1] with ...
            [0, 0, 2 * x[1,0], 0, 0, 2], # dy [0,2] with ...
            [0, 0, 0, 0, 0, 0], # dy [1,0] with ...
            [0, 0, 0, 0, 2 * x[2, 0], 0], # dy [1,1] with ...
            [0, 0, 0, 3, 0, 0], # similar
            [0, 0, 0, 0, 0, 0], # similar
            [0, 1, 0, 0, 0, 0], # similar
            [0, 2 * x[0,1], 0, 0, 0, 0] # similar
        ]
    )

    assert torch.equal(x.grad, (v.reshape(-1) @ dydx).reshape(x.shape))

    print('v: ', v)
    print('---------')
    print('v.reshape(-1): ', v.reshape(-1))
    print('---------')
    print('v.reshape(-1) @ dydx: ', v.reshape(-1) @ dydx )
    print('---------')
    print('(v.reshape(-1) @ dydx).reshape(x.shape) or dy/dx: ', (v.reshape(-1) @ dydx).reshape(x.shape))
    print('---------')

def cases_3():
    #Both x and y are matrix ( But there is not necessary to be compatible in size)
    x = torch.tensor([[1., 2.], [2., 3.], [1., 3.]], requires_grad=True) # 3 x 2

    # 1st dimension of y is not necessarily same as 1st dim of x
    y = torch.empty((4,3)) # 3 x 3
    y[0,0] = x[0,0] ** 2
    y[0,1] = x[1,0] + x[1,1]
    y[0,2] = x[1,0]**2 + 2 * x[2,1]
    y[1,0] = 0
    y[1,1] = x[2,0] ** 2
    y[1,2] = 3 * x[1,1] + 2
    y[2,0] = 0
    y[2,1] = x[0,1]
    y[2,2] = x[0,1]**2
    y[3,0] = x[0,0] + x[1,0] + x[2,0]
    y[3,1] = x[1,1] * x[2,1]
    y[3,2] = x[2,0] ** 2

    v = torch.tensor(
        [
            [1., 2., 3.],
            [2., 3., 1.],
            [1., 3., 2.],
            [1., 1., 1.]
        ]
    ) # v has to have the same dimension as x
    y.backward(v)

def cases_4():
    # Both x and y are vectors,
    # Test whether backward and grad are equivalent
    x = torch.tensor([1., 2.], requires_grad=True)

    y = torch.empty(3)
    y[0] = x[0] ** 2
    y[1] = x[0] ** 2 + 5 * x[1] ** 2
    y[2] = 3 * x[1]

    v = torch.tensor([1.0, 2.0, 3.0])
    y.backward(v)  # VJP.
    print('x.grad: ', x.grad)

    # dydx = grad(y, x, grad_outputs=v)
    # print('dydx: ', dydx)

def cases_5():
    # Both x and y are vectors
    # Test the retained graph
    # Second derivative

    x = torch.tensor([1., 2.], requires_grad=True)

    y = torch.empty(3)
    y[0] = x[0] ** 2
    y[1] = x[0] ** 2 + 5 * x[1] ** 2
    y[2] = 3 * x[1]

    v = torch.tensor([1.0, 2.0, 3.0])
    y.backward(v)  # VJP.

    print("y:", y)
    print("x.grad:", x.grad)


    # z = y[0] + 2 * y[1] + 3 * y[2] # These are equivalent to the above cases
    # z.backward() # Backward is implicit used for dim = 1
    # print('z:', z)
    # print('x.grad', x.grad)

    # Manual computation.
    # [
    #   [2, 0],
    #   [2, 20],
    #   [0, 3]
    # ]
    # dydx = torch.tensor(
    #     [
    #         [2 * x[0], 0],
    #         [2 * x[0], 10 * x[1]],
    #         [0, 3],
    #     ]
    # )
    #
    # assert torch.equal(x.grad, v @ dydx)

def cases_6():
    # Both x and y are vectors
    # Test the behavior of more than 2 layers
    x = torch.tensor([1., 2.], requires_grad=True)

    y = torch.empty(3)
    y[0] = x[0] ** 2
    y[1] = x[0] ** 2 + 5 * x[1] ** 2
    y[2] = 3 * x[1]

    v = torch.tensor([1.0, 2.0, 3.0])
    y.backward(v)  # VJP.

    print("y:", y)
    print("x.grad:", x.grad)

    # z = y[0] + 2 * y[1] + 3 * y[2] # These are equivalent to the above cases
    # z.backward() # Backward is implicit used for dim = 1
    # print('z:', z)
    # print('x.grad', x.grad)

    # Manual computation.
    # [
    #   [2, 0],
    #   [2, 20],
    #   [0, 3]
    # ]
    dydx = torch.tensor(
        [
            [2 * x[0], 0],
            [2 * x[0], 10 * x[1]],
            [0, 3],
        ]
    )

    assert torch.equal(x.grad, v @ dydx)

def cases_7():
    # Both x and y are vectors
    x = torch.tensor([1., 2.], requires_grad=True)

    y = torch.empty(3)
    y[0] = x[0] ** 2
    y[1] = x[0] ** 2 + 5 * x[1] ** 2
    y[2] = 3 * x[1]

    v = torch.tensor([1.0, 2.0, 3.0])
    y.backward(v)  # VJP.

    print("y:", y)
    print("x.grad:", x.grad)

    # z = y[0] + 2 * y[1] + 3 * y[2] # These are equivalent to the above cases
    # z.backward() # Backward is implicit used for dim = 1
    # print('z:', z)
    # print('x.grad', x.grad)

    # Manual computation.
    # [
    #   [2, 0],
    #   [2, 20],
    #   [0, 3]
    # ]
    dydx = torch.tensor(
        [
            [2 * x[0], 0],
            [2 * x[0], 10 * x[1]],
            [0, 3],
        ]
    )

    assert torch.equal(x.grad, v @ dydx)

# cases_1()
# cases_2()
# cases_3()
# cases_4()
cases_5()