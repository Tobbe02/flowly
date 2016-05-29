# Simple transformation chains in python.

flowly consists of `pipe`, a helper to write succinct transformation chains,
and `this`, a general proxy object to express transformations without
resorting to lambdas. A simple transformation of a pandas dataframe could look
like

```python
from flowly import this, pipe

+(pipe(df) |
  this.assign(foo=2 * this["LSTAT"] + this["NOX"]) |
  this.fillna(0) |
  this.drop_duplicates() |
  this.assign(foo_mean=this.groupby("RAD")["foo"].transform(np.mean)) |
  this[["foo", "foo_mean"]] |
  this.describe()
 )
```

Here, the pipe operator `|` passes the wrapped object to transformations and
the `+` operator extracts the results of the transformation chain. The
transformations can either be callables or objects that carry a
`_flowly_eval_(self, obj)` method that returns the result of the
transformation. The proxy object `this` refers to the current transformation
result and allows easy constructions of expressions that can be evaluated
inside the pipe.

Literal objects can be created with `lit`. Any literal object is itself a
proxy object and can be used to construct more complex expressions. In
particular literal callables allow to construct expressions involving evaluating
python functions inside more complex expressions. For example:

```python
    from flowly import this, pipe, lit

    +(pipe("a") |
      lit("{} ({})").format(this, lit(ord)(this))
     )
```

To inspect any expression the pretty function may be used:

```python
    from flowly import this, pretty

    print(pretty(this.foo() + this.bar))
```

Note: the submodules mpl/pd/py collect various utility functions for
matplotlib, pandas, general python respectively. However their interface is
still unstable and may change at any time.
