# Introduction

Pyburn, is simply an attempt to understand the infrastruucture that makes
up AI. There are currrently several frameworks like pytorch, tensorflow,
caffe, candle among others. The primary difference lies in their implementation,
how they handle memory, how tensors are pushed to the gpu and fundamentally the
language in which they are implemented in.

Now, i am **not** an expert in the subject and this is what prompted this exercise
and therefore whatever opinions i hold are falsifiable.

The current landscape of AI frameworks have fundamental similarities in that the data
 strcutures that are used to represent tensors and in the construct of neural network
graphs are similar.
Graph neural networks are prevalent and the tensors used are abstracted to hold types
of bool, int or floats. Quantization is offered in virtually all of them now, and the
ability to swap for precision types is also a common feature.

## Project layout

Pyburn follows the same methods of laying out modules as in tensorflow and pytorch.
It exposes the backends it suppports as can be seen in the first entry of its modules.
Currently the modules being supported are `ndarray` and `wgpu`. 

The ndarray backend is simply the cpu backend. The wgpu backend is supported to allow
inference in the integrated gpus present in platforms without dedicated gpus. It is
unadvisable to try using these backends to train fully fledged neural networks. It is
totally fine to perform inference in them though.

The top modules are:
    - `module`
    - `nn`
    - `lr_scheduler`
    - `optim`
    - `tensor`
    - `train`

`module` exposes the Module trait and it methods to support user defined layers.
The Module trait is needed for a layer to enable `autodiff`, and `fusion`.
These are the autodifferentiation and fusion properties in the modules.
Burn support backward autodifferentiation in its backends and as for fusion, it is
enabled by default if one uses the modules defined within the library.

There is a single class within module - Module - . This class is supposed to act as
a base class.

To define your own neural network layer;

```python
    Class LinearLayer(Module):
        ...

```

Honestly it is not properly defined to allow several methods to be hanged upon it
and allow the user-defined class to act as an outright neural network layer

The `tensor` module defines all the methods required for tensor operations to
be performed.
It exposes the TensorPy class and the Distribution enum.
Distribution enum specifies the probability distribution in which the tensors being
created are going to be instantiated.

The `nn` module is probably the largest in the package.
It has the following modules within it;
    attention, conv,loss, gru, transformer,pool

It has most of what one may need to build a neural network.

The rest of this document will be involved in elaborating various use cases.

