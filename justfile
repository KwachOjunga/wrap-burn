install:
    maturin develop

test: install
    python -m unittest -v tests\testing