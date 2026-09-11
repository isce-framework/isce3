## About Dense Offset Pairs

Dense offsets are calculated between selected pairs of scans in the stack rather than between every possible combination of acquisitions.

This is done as a systematic way to form pair networks for equally spaces nisar scans

For a chronologically ordered stack:

$$
S_0,\;S_1,\;S_2,\;S_3,\;S_4
$$

the parameter `pair_level` controls how far apart in the scan sequence two acquisitions may be when forming a dense-offset pair.

### `pair_level = 1`

Only adjacent scans are paired:

$$
(S_0,S_1),\;
(S_1,S_2),\;
(S_2,S_3),\;
(S_3,S_4)
$$

This gives the minimum set of pairwise measurements required to connect the stack. In this case, each scan offset is propagated directly along the date chain, and inversion provides no additional correction.

### `pair_level = 2`

Both adjacent scans and second-neighbor scans are paired:

$$
(S_0,S_1),\;
(S_1,S_2),\;
(S_2,S_3),\;
(S_3,S_4)
$$

and:

$$
(S_0,S_2),\;
(S_1,S_3),\;
(S_2,S_4)
$$

The additional pairs introduce redundant measurements into the scan network.

For example, the offset between $S_0$ and $S_2$ can now be constrained in two ways:

$$
S_0 \rightarrow S_1 \rightarrow S_2
$$

and directly by:

$$
S_0 \rightarrow S_2
$$

Ideally,

$$
d_{01}+d_{12}=d_{02}
$$

but in practice the dense-offset measurements contain correlation noise, so these values will not be exactly equal.

The redundant pair measurements reduce the influence of Ampcor fluctuations and measurement errors by jointly solving all available pairwise offsets to estimate a consistent offset for each scan relative to the stack reference.

## Dense Offset Inversion

The inversion process essentially derives, for each secondary scan, a dense offset to the reference that provides the **least-squares fit** to all calculated pairwise offset.

Suppose the stack contains four scans:

$$
S_0,\;S_1,\;S_2,\;S_3
$$

and $S_0$ is chosen as the reference scan.

Dense-offset matching does not initially give the offset of every scan relative to $S_0$. Instead, it gives offsets for selected scan pairs, for example:

$$
d_{01},\quad d_{12},\quad d_{23},\quad d_{02},\quad d_{13}
$$

where $d_{ij}$ is the measured range or azimuth offset between scans $S_i$ and $S_j$.

Define the unknown offsets relative to the reference as:

$$
x_0=0,\quad x_1,\quad x_2,\quad x_3
$$

Each pairwise measurement gives one linear constraint:

$$
x_j-x_i=d_{ij}
$$

For the example above:

$$
\begin{aligned}
x_1-x_0 &= d_{01} \\
x_2-x_1 &= d_{12} \\
x_3-x_2 &= d_{23} \\
x_2-x_0 &= d_{02} \\
x_3-x_1 &= d_{13}
\end{aligned}
$$

Since $x_0=0$, this becomes:

$$
\begin{bmatrix}
1 & 0 & 0 \\
-1 & 1 & 0 \\
0 & -1 & 1 \\
0 & 1 & 0 \\
-1 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
=
\begin{bmatrix}
d_{01} \\
d_{12} \\
d_{23} \\
d_{02} \\
d_{13}
\end{bmatrix}
$$

In compact form:

$$
Ax=d
$$

where:

- $A$ describes the scan-pair network.
- $d$ contains the measured pairwise dense offsets.
- $x$ contains the unknown offset of each scan relative to the reference scan.

Because several scan pairs constrain the same unknown offsets, the system is usually overdetermined. The solution is therefore obtained in a least-squares sense:

$$
\hat{x}
=
\arg\min_x \|Ax-d\|_2^2
$$

This is useful because the pairwise measurements are generally not perfectly consistent due to correlation noise.

For example:

$$
d_{01}+d_{12}
$$

may not exactly equal:

$$
d_{02}
$$

Even though both describe the displacement between $S_0$ and $S_2$.

The inversion finds the set of per-scan offsets that best satisfies all available pairwise measurements simultaneously.

For a standard least-squares problem, the solution can be written as:

$$
\hat{x}
=
(A^T A)^{-1}A^T d
$$

when $A^T A$ is invertible.

The same inversion is performed independently at every pixel in the dense-offset grid and separately for the two offset components:

- range offset
- azimuth offset

The output therefore gives, for each scan $S_i$, a range and azimuth offset field relative to the reference scan $S_0$:

$$
S_i
\rightarrow
\begin{cases}
\text{range offset relative to } S_0 \\
\text{azimuth offset relative to } S_0
\end{cases}
$$

These inverted offset fields are then combined with the coarse alignment during rubbersheeting and are used to produce the final resampled SLC stack.
