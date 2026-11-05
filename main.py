import pandas as pd

source = pd.DataFrame({
    "A": [0, 0, 1, 1, 0, 1, 2, 2],
    "B": [0, 0, 1, 1, 0, 1, 2, 2],
    "C": [1, 0, 1, 0, 1, 0, 2, 2]
})

target = source.rename(columns={
    "A": "X",
    "B": "Y",
    "C": "W"
})[["W", "X", "Y"]]

print("Source:")
print(source)

print("\nTarget:")
print(target)

import numpy as np 

def entropy(col):
    probs = col.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

for col in source.columns:
    print(f"Entropy of {col}: {entropy(source[col]):.4f}")
    joint_probs = source.groupby(["A", "B"]).size() / len(source)

def mutual_information(df, col1, col2):
    joint_probs = df.groupby([col1, col2]).size() / len(df)
    joint_entropy = -np.sum(joint_probs * np.log2(joint_probs))

    return entropy(df[col1]) + entropy(df[col2]) - joint_entropy

cols = source.columns
dep_matrix = np.zeros((len(cols), len(cols)))

for i, col1 in enumerate(cols):
    for j, col2 in enumerate(cols):
        dep_matrix[i, j] = mutual_information(source, col1, col2)

print("\nSource Dependency Matrix:")
print(np.round(dep_matrix, 4))

target_cols = target.columns
target_matrix = np.zeros((len(target_cols), len(target_cols)))

for i, col1 in enumerate(target_cols):
    for j, col2 in enumerate(target_cols):
        target_matrix[i, j] = mutual_information(target, col1, col2)

print("\nTarget Dependency Matrix:")
print(np.round(target_matrix, 4))


from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

# Eigen-decomposition
eig_s, vec_s = eigh(dep_matrix)
eig_t, vec_t = eigh(target_matrix)

# Sort eigenvectors by descending eigenvalues
vec_s = vec_s[:, np.argsort(eig_s)[::-1]]
vec_t = vec_t[:, np.argsort(eig_t)[::-1]]

# Umeyama's spectral similarity
similarity = np.abs(vec_t) @ np.abs(vec_s.T)
print("\nED Similarity Matrix:")
print(np.round(similarity, 4))

# Hungarian algorithm: maximize similarity
rows, cols = linear_sum_assignment(-similarity)

# Build permutation matrix: rows=target, columns=source
P = np.zeros_like(dep_matrix)
P[rows, cols] = 1

# Evaluate matching
cost = np.sum((P @ dep_matrix - target_matrix @ P) ** 2)

print("\nED Matching:")
for r, c in zip(rows, cols):
    print(f"{source.columns[c]} -> {target.columns[r]}")

print(f"ED Cost: {cost:.6f}")


from scipy.optimize import linprog

n = len(source.columns)
I = np.eye(n)

# Vectorized difference: X @ A_S - A_T @ X
K = np.kron(dep_matrix.T, I) - np.kron(I, target_matrix)

# Variables: entries of X and absolute-error variables t
c = np.r_[np.zeros(n*n), np.ones(n*n)]

A_ub = np.block([
    [ K, -I.repeat(n, axis=0).repeat(n, axis=1)],
    [-K, -I.repeat(n, axis=0).repeat(n, axis=1)]
])
b_ub = np.zeros(2*n*n)

# Each row and column of X must sum to 1
A_eq = np.zeros((2*n, 2*n*n))

for i in range(n):
    for j in range(n):
        A_eq[i, j*n + i] = 1
        A_eq[n+j, j*n + i] = 1

b_eq = np.ones(2*n)

result = linprog(
    c, A_ub=A_ub, b_ub=b_ub,
    A_eq=A_eq, b_eq=b_eq,
    bounds=(0, None),
    method="highs"
)

if not result.success:
    raise RuntimeError(result.message)

X_lp = result.x[:n*n].reshape((n, n), order="F")

print("\nLP Relaxed Matrix:")
print(np.round(X_lp, 4))
print(f"LP objective (L1): {result.fun:.6f}")


lp_rows, lp_cols = linear_sum_assignment(-X_lp)

P_lp = np.zeros((n, n))
P_lp[lp_rows, lp_cols] = 1

lp_cost = np.sum((P_lp @ dep_matrix - target_matrix @ P_lp) ** 2)

print("\nLP Matching:")
for r, c in zip(lp_rows, lp_cols):
    print(f"{source.columns[c]} -> {target.columns[r]}")

print(f"LP Cost: {lp_cost:.6f}")


from scipy.optimize import minimize

def qp_objective(x):
    X = x.reshape(n, n)
    diff = X @ dep_matrix - target_matrix @ X
    return np.sum(diff ** 2)

constraints = [
    {"type": "eq",
     "fun": lambda x: x.reshape(n, n).sum(axis=1) - 1},
    {"type": "eq",
     "fun": lambda x: x.reshape(n, n).sum(axis=0) - 1}
]

result_qp = minimize(
    qp_objective,
    x0=np.ones(n * n) / n,
    method="SLSQP",
    bounds=[(0, 1)] * (n * n),
    constraints=constraints,
    options={"ftol": 1e-12}
)

if not result_qp.success:
    raise RuntimeError(result_qp.message)

X_qp = result_qp.x.reshape(n, n)

print("\nQP Relaxed Matrix:")
print(np.round(X_qp, 4))
print(f"QP Objective: {result_qp.fun:.6f}")

qp_rows, qp_cols = linear_sum_assignment(-X_qp)

P_qp = np.zeros((n, n))
P_qp[qp_rows, qp_cols] = 1

qp_cost = np.sum(
    (P_qp @ dep_matrix - target_matrix @ P_qp) ** 2
)

print("\nQP Matching:")
for r, c in zip(qp_rows, qp_cols):
    print(f"{source.columns[c]} -> {target.columns[r]}")

print(f"QP Cost: {qp_cost:.6f}")


from itertools import combinations

def matching_cost(P):
    diff = P @ dep_matrix - target_matrix @ P
    return np.sum(diff ** 2)


def hill_climbing(P):
    current = P.copy()
    current_cost = matching_cost(current)

    while True:
        best = current.copy()
        best_cost = current_cost

        for i, j in combinations(range(n), 2):
            candidate = current.copy()
            candidate[[i, j], :] = candidate[[j, i], :]

            cost = matching_cost(candidate)

            if cost < best_cost - 1e-10:
                best = candidate
                best_cost = cost

        if best_cost >= current_cost - 1e-10:
            break

        current = best
        current_cost = best_cost

    return current, current_cost


P_start = np.eye(n)
P_hc, hc_cost = hill_climbing(P_start)

print("\nHC Matching:")
for r, c in zip(*np.where(P_hc == 1)):
    print(f"{source.columns[c]} -> {target.columns[r]}")

print(f"HC Cost: {hc_cost:.6f}")


best_cost = float("inf")
best_P = None
visited = 0
pruned = 0

def branch_and_bound(mapping, remaining):
    global best_cost, best_P, visited, pruned

    k = len(mapping)

    # Lower bound from already matched nodes
    lb = 0.0
    for i in range(k):
        for j in range(k):
            lb += (
                dep_matrix[i, j]
                - target_matrix[mapping[i], mapping[j]]
            ) ** 2

    if lb >= best_cost - 1e-10:
        pruned += 1
        return

    if k == n:
        visited += 1
        best_cost = lb
        best_P = np.zeros((n, n))

        for i, target_idx in enumerate(mapping):
            best_P[target_idx, i] = 1
        return

    for target_idx in remaining:
        branch_and_bound(
            mapping + [target_idx],
            [j for j in remaining if j != target_idx]
        )


branch_and_bound([], list(range(n)))

print("\nBranch and Bound Matching:")
for r, c in zip(*np.where(best_P == 1)):
    print(f"{source.columns[c]} -> {target.columns[r]}")

print(f"B&B Cost: {best_cost:.6f}")
print(f"Complete permutations evaluated: {visited}")
print(f"Branches pruned: {pruned}")