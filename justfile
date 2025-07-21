# use a shell of your own liking
set shell := ["powershell.exe", "-c"]

alias i := install

install:
    maturin develop

test: install
    python -m unittest -v tests\testing