# AutoDiff

This repo try to explain the behavior of torch.autograd package
- retain graph is useful when we have two or more loss functions. Specifically, we want to accumulate the partial derivative in the shared nodes between 2 losses. 
- create_graph is useful when we want to compute the higher order derivative.
  - Note: If we have one loss, retain_graph and create_graph tend to behave similarly.
- backward: Computes the vector Jacobian product. Notes that the function implicitly assumes that the input is a scalar
- grad: Similar to backward function

## References
[Explain retain_graph](https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method)
[Reverse-mode automatic differentiation](https://sidsite.com/posts/autodiff/): The blog provides the code from scratch of Variable function in Pytorch
[The chain rule, Jacobians, autograd, and shapes](https://heiner.ai/blog/2023/02/19/chain-rule-jacobians-autograd-shapes.html): The blogs explains very good the concept of vector-Jacobian. It also shows the difference between the JAX and Pytorch when deal with backprop