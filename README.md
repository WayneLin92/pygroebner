# pygroebner

This is a package for Groebner basis over prime 2.

The `pygroebner` module can be used to
* create an algebra/DGA over F2 by generators and relations;
* calculate the generators of an ideal of annihilators;
* calculate the subalgebra generated by some elements;
* export the algebra/DGA to latex;
* save the algebra/DGA in an sqlite3 database;
* load the algebra/DGA in an sqlite3 database.

In the release version of the package, the module will be able to
* compute the homology of a DGA;
* export the algebra to an HTML file so that you can visualize and interact with it.

The sqlite3 database is intended to be used as a compact form to be shared among people.
It also serve as the interface to my C++ groebner basis project, which runs much faster
but is not suitable for a casual use when the computation is not super heavy.

## Usage
```python
>>> from pygroebner import new_alg, load_alg
>>> A = new_alg(key_mo="Lex")
>>> x = A.add_gen("x", (1, 1))
>>> y = A.add_gen("y", (1, 1))
>>> A.add_rel(x * x + y * y)
>>> A.add_rel(y ** 3)
>>> x ** 2
y^2
>>> print(A.latex_alg())
\section{Gens}

$x$ (1, 1)

$y$ (1, 1)


\section{Relations}

$x^2 = y^2$

$y^3 = 0$

>>> A.save_alg("tmp.db", "A")
>>> B = load_alg("tmp.db", "A")
>>> B.gen['x'] * B.gen['y']
xy
>>> B.gens == A.gens
True
```


