The following crate exposes the modules in burn i.e {backend, nn, config, train  .etc}
as wrappers so that they can be used internally by any crate/application intending to
expose its function in Python.


`wrap-burn`| `pyburn` began as an effort to create a python package with a Rust 
implementation of a known [deep learning](https://github.com/tracelai/burn) library;
so that it may be used expressly as a direct replacement of pytorch.

I am seizing this effort as of today.

First let me say that burn and its family of crates is a well crafted effort with 
clear guidelines on the engineering tradeoffs it has made in order to make it a success. 
It is a lightweight project in comparison to pytorch and has delivered on its promise
to permit individuals to build neural network architectures targeting several backends.
It cannot be overstated how useful such an undertaking is.

As to why the effort is coming to a stand still.
---

`burn` is built to target an end user application. ie, its design aims at being built 
for a specific application that is later to be deployed to a targeted setting.
This has the implications that it takes up an approach that ensures the overall project
is tailored for a singular application and not meant to explore the entire design space
that AI applications occupy.[1]
This means that the entire project is built to alleviate pressure on development time 
and ensure a lightweight project.
All this is to say that burn is not built to be used as an intermediate layer to a package
that is to serve as a library.

I have rather limited experrience building libraries but there are a few things that i 
have picked up.

- The function signature defines the contract between your library and the users of said 
    library. This signature is not limited to functions; It is determined by the 
    implementation of your classes, in Rust's cases structs and enums; it is also defined
    by the behavior of the objects that are instantiated. In Rust, this said behavior is 
    captured fundamentally by the traits that offer methods that allow interoperability 
    between objects that differ in structure.
- This contract establishes the degrees of freedom a user has with what one's library proffers.

I will try to get into as much technical detail as possible; i only hope i can articulate
it efficiently.

In the case of burn, its target spans a vast application space and to achieve this it has 
had to develop in-house tools to permit it to perform its function.
It relies on a crate called [cubecl](https://github.com/tracelai/cubecl) which is a computing
platform that permits it to be generic across several backends.ie vulkan, DirectX, Cuda, webgpu,
and probably one more i am forgetting.
These inhouse tools are a selection of compilers that permit its compilation to these different
target architectures. To enhance developer experience burn opted in to a Rust feature that lowers
compile time while effectively locking it out of essentially having properties that are defined 
at runtime. This enhances developer experience but prevents it from being used as a library. 
Burn is the final platform upon which a user can build upon.
If you think about it; this means burn enforces its use to the Rust ecosystem; though one could 
argue that it can be bundled to a dynamic library and used in other languages through their 
foreign function interfaces. I'm uncertain about this though.

 `const generics`
 ---
These have been a recent bane in my existence. The foundation of any deep learning module is 
dependent on the structure of it Tensor. All data that is to be acted upon is held in a tensor
and all computations that are to be carried out on the GPU rely on the tensor that establishes
how tiling operations will be performed and it also which operations are viable to be performed
on the tensor depending on its datatype.
This is to say that a Tensor is a generic data structure that can hold an N-dimensional array of
values whose data type is an integer, float or boolean type. (Its pointless to get into the exact
data types ie. u32, f32).
# In burn's case (and also in pytorch) in order to load a tensor with data whose shape is unkown.

How they work
---

example implementations
```rust
```

```rust
```

Preferred alternatives
---





Footnotes
---
[1] On the nature of AI applications - exploring the complexity that AI systems place on their design.