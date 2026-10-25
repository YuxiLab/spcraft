# Numerical examples to trace before coding

These are small arithmetic examples, not implementation code. Predict each
intermediate result, then compare it with the tables. All indices are zero-based.
The optional checker uses exact fractions and does not implement an LU solver.

## 1. Basic elimination, fill, and triangular solves

Use the nonsymmetric, strictly row diagonally dominant matrix

```text
    [ 4 1 1 ]         [ 1 ]       [  9 ]
A = [ 2 5 0 ],  x* = [ 2 ],  b = [ 12 ].
    [ 0 1 6 ]         [ 3 ]       [ 20 ]
```

The first pivot is 4. The multiplier L(1,0)=1/2 changes row 1 to
`[0,9/2,-1/2]`. U(1,2) is fill: A(1,2) was absent. The next multiplier is
`L(2,1)=1/(9/2)=2/9`, leaving the last pivot `6-(2/9)*(-1/2)=55/9`.

```text
    [  1   0   0 ]       [ 4  1    1   ]
L = [ 1/2  1   0 ],  U = [ 0 9/2 -1/2  ].
    [  0  2/9  1 ]       [ 0  0   55/9 ]
```

Multiply L*U yourself. In particular, `(L*U)(1,2)=1/2-1/2=0` reconstructs
the original missing entry even though U(1,2) is nonzero.

For `L*y=b`, obtain `y={9,15/2,55/3}`. Back substitution gives
`x[2]=3`, `x[1]=(15/2+3/2)/(9/2)=2`, and `x[0]=(9-2-3)/4=1`.

### CSC representation

```text
A.col_ptr = {0,2,5,7}
A.row_id  = {0,1, 0,1,2, 0,2}
A.val     = {4,2, 1,5,1, 1,6}

L.col_ptr = {0,2,4,5}
L.row_id  = {0,1, 1,2, 2}
L.val     = {1,1/2, 1,2/9, 1}

U.col_ptr = {0,1,3,6}
U.row_id  = {0, 0,1, 0,1,2}
U.val     = {4, 1,9/2, 1,-1/2,55/9}
```

Pointers delimit columns; their differences count entries. The U column-2
pattern includes row 1 although A column 2 has only rows 0 and 2. The symbolic
plan supplies this extra slot before numerical factorization starts.

### Left-looking accumulator trace

For column j, after processing earlier columns up to k-1, the unfinalized rows
of the accumulator represent the corresponding entries of
`A(:,j) - sum_{q<k} L(:,q)*U(q,j)`. In this algorithm an extracted U value at
row k stays in its accumulator slot; updates traverse only L's strict lower
part. Thus the invariant applies to the still-unfinalized rows, not all rows
after U values have been extracted.

| Column | Action | Accumulator values at rows 0,1,2 |
|---|---|---|
| 0 | Scatter A(:,0) | `{4,2,0}` |
| 0 | No earlier column; extract pivot/multipliers | U(0,0)=4, L(1,0)=1/2 |
| 1 | Scatter A(:,1) | `{1,5,1}` |
| 1 | Extract U(0,1)=1; apply strict L(:,0) | `{1,9/2,1}` |
| 1 | Extract pivot and multiplier | U(1,1)=9/2, L(2,1)=2/9 |
| 2 | Scatter A(:,2) | `{1,0,6}` |
| 2 | Extract U(0,2)=1; apply strict L(:,0) | `{1,-1/2,6}` |
| 2 | Extract U(1,2)=-1/2; apply strict L(:,1) | `{1,-1/2,55/9}` |

Row 1 becomes touched through fill in the last column. Record it for cleanup
even though it was absent from the original input column. A structurally stored
zero U coefficient still needs a slot and can still require planned traversal.

### Permutation experiment

With `order={2,0,1}`, `inverse_order={1,2,0}`. Both rows and columns are reordered:

```text
    [ 6 0 1 ]         [ 20 ]       [ 3 ]
B = [ 1 4 1 ], rhs = [  9 ], z* = [ 1 ].
    [ 0 2 5 ]         [ 12 ]       [ 2 ]
```

Check `B*z*=rhs`. The solution scatter is `x[2]=z[0]`, `x[0]=z[1]`,
`x[1]=z[2]`. The L/U factors above belong to A in identity order; they cannot
be reused as the factors of B. Reordering requires a compatible plan and a new
numerical factorization.

## 2. One dense cross: diagonal LU and two triangular solves

Consider a destination with ordered columns `J={2,3}`, lower rows `R={4}`,
and upper columns `C={5}`. Its already-updated cross is

```text
A11 = [4 2]    A21 = [4 8]    A12 = [2]
      [2 4]                        [2]
```

Factor A11, then solve the two equations from the block factorization:

```text
L11 = [ 1  0]   U11 = [4 2]   L21 = [1 2]   U12 = [2]
      [1/2 1]         [0 3]                       [1]

L21*U11 = A21           L11*U12 = A12
```

For `L21*U11=A21`, the first unknown is `4/4=1`, and the second is
`(8-1*2)/3=2`: a right upper-triangular solve proceeds in increasing unknown
column order. For the left unit-lower solve, U12's first value is 2, then its
second value is `2-(1/2)*2=1`.

Column-major output buffers are

```text
diagonal = {4,1/2, 2,3}   // strict-lower L11 and upper U11 packed together
lower    = {1,2}         // leading dimension |R|=1
upper    = {2,1}         // leading dimension |J|=2
```

The L diagonal is implicit in the packed buffer: its stored diagonal values 4
and 3 belong to U. With `pivot_tolerance=3`, the second pivot fails at global
ordered column 3; successful output must not be published.

## 3. A complete nonsymmetric example with different row and column maps

This example is supplied by `MakeNonsymmetricMatrix` in
[`LuFixture.h`](../../tests/solver/LuFixture.h). In ordered coordinates:

```text
    [8  0   2     0   ]       [ 1   0    0   0]
B = [2 10  1/2    3   ]   L = [1/4  1    0   0]
    [0  2  12    8/5  ]       [ 0  1/5   1   0]
    [4  0   4   65/4  ]       [1/2  0   1/4  1]

    [8  0  2  0]
U = [0 10  0  3]
    [0  0 12  1]
    [0  0  0 16]
```

B is strictly row diagonally dominant and has a nonsymmetric sparsity pattern.
The planned U(1,2) slot cancels to zero. The plan preserves it.

| Panel | J | R | C | Contributors |
|---|---|---|---|---|
| 0 | `{0}` | `{1,3}` | `{2}` | none |
| 1 | `{1}` | `{2}` | `{2,3}` | `{0}` |
| 2 | `{2,3}` | empty | empty | `{0,1}` |

For panel 0, assembly gives `A11={8}`, `A21={2,4}`, and `A12={2}`.
Factoring gives `L21={1/4,1/2}`, `U12={2}`. The second lower-buffer element
represents global row 3, not global row 2.

Panel 1's original upper cross is `{1/2,3}`. Panel 0 subtracts
`(1/4)*{2,0}`, producing `{0,3}`. Its diagonal and lower cross remain 10 and 2.
Factoring gives L(2,1)=1/5.

Panel 2 assembles and receives these diagonal-cross contributions:

```text
original A11       source 0         source 1        updated A11
[12  8/5 ]    -    [0 0]      -     [0 3/5]     =   [12   1  ]
[ 4 65/4 ]         [1 0]            [0  0 ]         [ 3 65/4 ]
```

Its packed diagonal becomes `{12,1/4,1,16}`. The direct update test supplies
the completed source factors by hand, so a failing `UpdatePanel` can be diagnosed
before `Factorize` works. Another test exercises all three destination blocks
with nonconsecutive source maps and an absent source coupling.

For identity order and `x*={1,2,3,4}`, the RHS is `{14,71/2,232/5,81}`.
For the permuted fixture, the helper constructs original A from this same B
using `order={2,0,3,1}`. Build `b=A*x*` in original coordinates; do not reuse the
identity-order RHS. The expected factors in ordered coordinates stay the same.

## 4. Dependencies and the boundary of the learning model

The example above has edges `0->1`, `0->2`, `1->2`. These three panels form a
dependency chain despite one vertex having two outgoing edges. More workers
cannot remove that critical path. The separate fork/join test uses
`0->2<-1`, where panels 0 and 1 can run independently.

Compare the three policies on `[[0,1],[1,0]]`: the assignment rejects the first
pivot; a row swap permits LU; replacing the zero diagonal changes the numerical
problem being factorized. Keep that distinction when reading the library
[pivoting lesson](source-reading.md#lesson-g-pivoting-and-numerical-stability).

Return to the [sequence](README.md) and implement the dense
triangular solves and LU first. Use [checkpoints](checkpoints.md) to run one
exercise at a time, then translate the same arithmetic to sparse storage.
