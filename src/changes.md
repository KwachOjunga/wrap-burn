# Revisions made `pyburn v0.1.7`

- Migrated modules to nn, tensor and train to their separate modules.


- On the nn module ;
    you cannot eliminate the common_nn_exports module
    all else is replacable
    - Yes, so this was a flawed requirement

### Ticket1 20/09/25
- 
There is a fundamental flaw in the common_nn_tensor_exports module 
in the root tensor module. While the types defined in it require
no backend initialization, their return types are mostly tensors
of `PyResult<TensorPy>` types. This introduces a quagmire of sorts
since this requires reimplementation of that specific module for
both the currently supported backends.

