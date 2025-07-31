`PyBurn`
---

`wrap-burn` | `pyburn` began as an effort to create a python package with a Rust 
implementation of a known [deep learning](https://github.com/tracelai/burn) library;
so that it may be used expressly as a direct replacement of pytorch.

I am now trying to learn how burn would be used as a foundation for a deep learning
library in a dynamic language.


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
All this is to say that burn is not built to be used as an intermediate layer to a 
package that is to serve as a library.

I have rather limited experrience building libraries but there are a few things that i 
have picked up.

- The function signature defines the contract between your library and the users of 
    said library. This signature is not limited to functions; It is determined by the 
    implementation of your classes, in Rust's cases structs and enums; it is also 
    defined by the behavior of the objects that are instantiated. In Rust, this said
    behavior is captured fundamentally by the traits that offer methods that allow 
    interoperability  between objects that differ in structure.
- This contract establishes the degrees of freedom a user has with what one's library
    proffers.

I will try to get into as much technical detail as possible; i only hope i can articulate
it efficiently.

In the case of burn, its target spans a vast application space and to achieve this it has 
had to develop in-house tools to permit it to perform its function.
It relies on a crate called [cubecl](https://github.com/tracelai/cubecl) which is a 
computing platform that permits it to be generic across several backends.ie vulkan, 
DirectX, Cuda, webgpu, and probably one more i am forgetting.
These inhouse tools are a selection of compilers that permit its compilation to these 
different target architectures. To enhance developer experience burn opted in to a Rust
feature that lowers compile time while effectively locking it out of essentially having 
properties that are defined at runtime. This enhances developer experience but prevents 
it from being used as a library. 
Burn is the final platform upon which a user can build upon.
If you think about it; this means burn enforces its use to the Rust ecosystem; though 
one could argue that it can be bundled to a dynamic library and used in other languages 
through their foreign function interfaces. I'm uncertain about this though.

 `const generics`
 ---
These have been a recent bane in my existence. The foundation of any deep learning module
is dependent on the structure of it Tensor. All data that is to be acted upon is held in 
a tensor and all computations that are to be carried out on the GPU rely on the tensor 
that establishes how tiling operations will be performed and it also which operations are
viable to be performed on the tensor depending on its datatype.
This is to say that a Tensor is a generic data structure that can hold an N-dimensional 
array of values whose data type is an integer, float or boolean type. (Its pointless to 
get into the exact data types ie. u32, f32).
# In burn's case (and also in pytorch) in order to load a tensor with data whose shape is unkown.


# What are the current mechanics of burn's implementation and how do they place a limitation 
# on how a python package can be built from it.

How they work
---
So const generics are a feature that neccessitate the definition of a generic parameter at 
compile time. So basically, given a struct Foo whose parameter is A, to convert that 
parameter to a const type, you append the const keyword to the type.
The use of a const in such an instance requires the explicit definition of the type.


```rust
    struct Foo<A>{}

```
code_1.1:

```rust
    struct Foo<const A: u32>{}

```
code_1.2


The const only have the builtin types for its type and also in their implementation they 
cannot have the given parameter as a field in their type.

```rust
struct Foo<const A: u32>{ inner: A}
```
code_1.3:

In burn's case; it's use of a const parameter is in its Tensor data structure.

```rust
struct Tensor<B, const D: usize, K = Float>
where B: Backend
{
    primitive: K::Primitive
}
```
code_1.4: Burn's Tensor data structure.

This simple feature has massive ramifications.
It means in order to use the Tensor data structure; the dimensions of the data must be 
known at compile time; meaning the Tensor type cannot be expected to be used even in 
place of an expected trait object.
i.e,
Imagine a wrapper for a Tensor data structure;

```rust
use pyo3::prelude:*;


#[pyclass]
struct TensorPy{
    inner: Box<&dyn Backend>
}
```
code_1.5: Example TensorPy struct that is to be used to expose the internal `Tensor` struct

Such an approach is required since pyo3 is incapable of exposing a type that has 
generic parameters; So one needs to instantiate such a wrapper to act as a layer of 
indirection. The inner type is a Box type that serves as another indirection layer
(zero-cost abstractions right??),to offer a wide pointer that attempts to encapsulate
the dynamic size of the Tensor (since Tensors are n-dimensional).
This approach primarily aims at gaining the flexibility of specifying various types of
Tensor via a single interface.From this it follows that one can reconstruct various 

                    ```rust 
                    Tensor<B,D,K>
                    ``` 
methods to the wrappers interface and expose them to the python interface.

```rust 

#[pymethods]
impl Tensorpy {

    #[staticmethod]
    fn all(&self) -> Self {
        let tensor = self.inner.downcast::<Tensor<Wgpu, 1>>();
        Into::<TensorPy>::into(tensor.all())
    }
}

```
code_1.6: Applying internal methods to a wrapper type


Even from this construct it's easy to spot how implementing this turns out to be an
intractable problem. i.e 
the downcast method on the Box type neccessitates that its concrete type be given;
This means one has to specify an all method for all Tensors of every imaginable dimension.
That's the utility of generics for you!

The second line of conversion into TensorPy requires that the From trait is also implemented 
on every type of Tensor for conversion to be seamless.

```rust

impl From<Tensor<Wgpu,1>> for TensorPy{
    fn from (other: Tensor<Wgpu, 1>) -> Self{
        Self{
            inner: Box::new(other)
        }
    }
}

impl From<Tensor<Wgpu,2>> for TensorPy{
    fn from (other: Tensor<Wgpu, 1>) -> Self{
        Self{
            inner: Box::new(other)
        }
    }
}

impl From<Tensor<Wgpu, 100 >> for TensorPy{
    fn from (other: Tensor<Wgpu, 1>) -> Self{
        Self{
            inner: Box::new(other)
        }
    }
}

// This can go on ~forever.

```
code_1.7: Implementing From trait to permit conversions between Rust and Python types


This is only with respect to a change in tensor dimensions; God forbid you have a 
tensor with a different datatype.

But for those who are familiar with Burn; it is obvious that the Backend trait is **not**
object safe. Therefore the Tensor struct cannot be used as a trait object even with an
indirection type as `Box<T>`.

So this implementation is wrong.

example implementations
```rust
```

```rust
```



# What does it take to build a deep learning library 

Preferred alternatives
---





Footnotes
---
[1] On the nature of AI applications - exploring the complexity that AI systems place on their design.