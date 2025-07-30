Current Issue (17/07/2025)
---

Building a wrapper around a generic struct to a dynamic language that does not use similar
semantics around generics.

One of the the fundamental properties around generics in Rust is that they fall into two known
categories; they either end up being statically dispatcged or dynamically dispatched.
This duality is enabled to allow for flexibility around runtime and compile time requirements.i.e
Static dispatch allows for optimizations in the compile where the generic type is converted to a 
non-generic type relying on each instance of the used generic and dynamic dispatch yields a trait 
object that is akin to an obscure type that basically ends up expecting a type that implements a 
particular trait at runtime.

This is the crux of the issue.
In order to create a wrapper around a generic type; for a library as pyo3, one is constrained to using
macros in their building process, whereby, the wrapper can be a struct containing a field to the 
required type.
An alternative to this is to use empty structs and basically implement methods that use the desired 
type. Implementing a struct whose generic requirements is a type is slightly straightforward compared to
one whose parametres require some sort of specification beforehand; ie a particular value to be offered 
at runtime.


```rust
 struct Tensor<B: Backend, const D = usize, const K = Float> {}
```
ext1: Generic type with multiple parametres as requirements

```rust
struct BatchNorm<B: Backend> {}
```
ext2: Generic type with a single requirement for its parameter


In ext1, building a wrapper while retaining its generic capabilities can prove difficult since this is a 
struct that relies on compile time definitions. This means the wrapper will end up nullifying the generic 
nature of the type if its parameters are to be defined at compile time.


```rust
 struct PyTensor {
    pub inner: Tensor<Wgpu, 1>
 }
 ```
ext3: Nullification of the generic abilities of Tensor struct


To workshop a few alternatives of this;
 requirements ; - usize and K which may or may not be defined.

```rust
#[derive(PartialEq,PartialOrd)]
enum TenNum {
     One = 1, Two = 2, Three = 3, Four = 4
}

fn create_tensor(val: usize) -> Tensor<Wgpu> {
    match val {
        TenNum::One => { Tensor<Wgpu, 1>},
        TenNum::Two => Tensor<Wgpu, 2>,
        TenNum::Three => Tensor<Wgpu, 3>,
        TenNum::Four => Tensor<Wgpu, 4>
    }
}
```

OK, so whatever i had in mind basically doesn't work

Considerations;
- Combining traits into single traits to lower size of pointer to whatever reference; i.e
    ```rust
    trait Hei {
        fn name(&self);
        fn jump(&self;
    }

    trait HeiAsRef: Hei + AsRef<T> {}

    // application 
    fn foo(s: impl HeiAsRef) -> &'static str {...}
    
    ```

    

----
Challenges
1. In order to access methods implemented by structs within the wrappers and expose them 
    as default methods of their wrappers, alot of conversion has to occur; but this is Rust,
    Zero-cost abstractions right??

2. Some of the methods implemented for the field structs of the wrappers require ownership 
    i.e Self has to be a paramenter in order to perform in-place mutation; this clashes with 
    the need for Pyo3 types to be shared references for exposure to the Python interface; 
    In order to use said methods there has to be a lot of cloning. -- 
    Perhaps there is a type; implements Copy trait, offers interior mutability and can be used 
    safely in a concurrent setting.

3. Almost all types in Burn are generic; and this is to allow it to offer a single abstraction across
    multiple backends; The issue that arises is that in order to expose types in dynamic language, 
    the implementations ought to be concrete. This raises a connundrum whereby methods have to be defined
    for every concrete implementation of a type that is realised from the generic type.