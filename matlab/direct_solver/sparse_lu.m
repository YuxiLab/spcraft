%% Sparse LU: build one factor column at a time
% Run this script to generate a random sparse matrix, trace the factorization,
% check A = L*U, and visualize fill-in. Local functions appear at the end.
%
% Mathematics: A = L*U, with L unit lower triangular and U upper triangular.
% At column k, start with v = A(:,k). For contributing earlier columns j:
%   U(j,k) = v(j);
%   v(j+1:n) = v(j+1:n) - L(j+1:n,j)*U(j,k).
% Finish with U(k,k) = v(k) and L(k+1:n,k) = v(k+1:n)/U(k,k).
% The updates look LEFT at columns already computed, hence "left-looking".
% MATLAB indices start at 1 here; the lecture's C++ indices start at 0.
% Reference API: https://www.mathworks.com/help/matlab/ref/lu.html

%% 1. Create a reproducible nonsymmetric sparse matrix
n = 10;
density = 0.20;
seed = 7;
show_steps = true;
show_plots = true;
rng(seed, 'twister');

R = sprandn(n, n, density);
R = R - spdiags(diag(R), 0, n, n);
% Strict row diagonal dominance makes every leading principal block
% nonsingular, so the teaching factorization needs no row swaps.
diagonal_margin = 1;
d = sum(abs(R), 2) + diagonal_margin;
A = R + spdiags(d, 0, n, n);

fprintf('Random %d-by-%d matrix: seed=%d, nnz(A)=%d\n', n, n, seed, nnz(A));
fprintf('The diagonal is strengthened to make no-pivot LU applicable.\n');
if show_steps
    disp('A =');
    disp(full(A));
end

%% 2. Compute LU using the algorithm, not a call to lu
[L, U] = LeftLookingLU(A, show_steps);

%% 3. Check the factorization and solve A*x = b
factor_error = norm(A - L*U, 'fro') / norm(A, 'fro');
x_exact = randn(n, 1);
b = A*x_exact;
y = L \ b;                 % Forward substitution: L*y = b.
x = U \ y;                 % Back substitution: U*x = y.
solve_residual = norm(A*x - b) / norm(b);
solution_error = norm(x - x_exact) / norm(x_exact);

% Independent reference: MATLAB's sparse LU can reorder rows and columns.
% Four outputs satisfy P*A*Q = L_ref*U_ref (no scaling matrix requested).
[L_ref, U_ref, P, Q] = lu(A);
reference_factor_error = norm(P*A*Q - L_ref*U_ref, 'fro') / norm(A, 'fro');
x_ref = Q * (U_ref \ (L_ref \ (P*b)));
reference_solution_error = norm(x - x_ref) / norm(x_ref);

fprintf('\nVerification\n');
fprintf('  ||A - L*U||_F / ||A||_F       = %.3e\n', factor_error);
fprintf('  ||A*x - b||_2 / ||b||_2       = %.3e\n', solve_residual);
fprintf('  ||x - x_exact|| / ||x_exact|| = %.3e\n', solution_error);
fprintf('  Reference factor error        = %.3e\n', reference_factor_error);
fprintf('  Difference from reference x   = %.3e\n', reference_solution_error);

% The small, diagonally dominant demonstration is well conditioned.
check_tolerance = 1e-11;
assert(factor_error < check_tolerance, 'The computed factors do not reconstruct A.');
assert(solve_residual < check_tolerance, 'The computed solution has a large residual.');
assert(solution_error < check_tolerance, 'The computed solution differs from x_exact.');
assert(reference_factor_error < check_tolerance, 'The reference factorization check failed.');
assert(reference_solution_error < check_tolerance, 'The solution differs from MATLAB LU.');

%% 4. Show how elimination creates new nonzeros
% Compare PATTERNS, not L+U values: numerical addition could cancel entries.
factor_pattern = spones(spones(L) + spones(U));
fill_pattern = factor_pattern - factor_pattern .* spones(A);
fprintf('\nStructure\n');
fprintf('  nnz(L)=%d, nnz(U)=%d, combined factor positions=%d\n', ...
    nnz(L), nnz(U), nnz(factor_pattern));
fprintf('  New factor positions absent from A (fill-in): %d\n', nnz(fill_pattern));
fprintf('  The L and U diagonals occupy the same positions in the combined pattern.\n');
fprintf('  General solvers use pivoting for stability and ordering to reduce fill.\n');
fprintf('  This teaching code uses dense scratch and sparse insertion, not production timing.\n');

if show_plots
    figure('Name', 'Sparse LU: input, factors, and fill-in', 'Color', 'w');
    subplot(2, 3, 1); spy(A); title('Input A');
    subplot(2, 3, 2); spy(L); title('Unit lower factor L');
    subplot(2, 3, 3); spy(U); title('Upper factor U');
    subplot(2, 3, 4); spy(factor_pattern); title('Combined factor pattern');
    subplot(2, 3, 5); spy(fill_pattern); title('New positions: fill-in');
    subplot(2, 3, 6); spy(P*A*Q); title('Reference reordered input P*A*Q');
end

%% Local teaching routines
function [L, U] = LeftLookingLU(A, show_steps)
% Assumptions: square nonsingular input and no zero pivots in natural order.
% This example's strict diagonal dominance guarantees those conditions.
    n = size(A, 1);
    L = speye(n);
    U = spalloc(n, n, nnz(A));

    for k = 1:n
        % SYMBOLIC: L(i,j) creates a dependency j -> i. Only earlier
        % columns are traversed; future columns of L are still identity.
        seeds = find(A(1:k-1, k));
        contributors = ReachEarlierColumns(L, seeds, k);

        % NUMERIC: the dense scratch vector makes scattered updates simple.
        v = full(A(:, k));
        if show_steps
            fprintf('\nColumn k=%d: seeds=%s, contributing columns=%s\n', ...
                k, mat2str(seeds.'), mat2str(contributors.'));
        end

        for j = contributors.'
            ujk = v(j);
            U(j, k) = ujk;
            % A structurally possible contribution can cancel to zero.
            if ujk == 0
                continue;
            end
            [local_rows, ~, lij] = find(L(j+1:n, j));
            rows = local_rows + j;
            v(rows) = v(rows) - lij*ujk;
            if show_steps
                fprintf('  Use column j=%d: U(j,k)=% .6g, update rows %s\n', ...
                    j, ujk, mat2str(rows.'));
            end
        end

        % SPLIT: the residual above k is U; below k it becomes L after scaling.
        pivot = v(k);
        if pivot == 0
            error('sparse_lu:ZeroPivot', 'Zero pivot at column %d; row pivoting is required.', k);
        end
        U(k, k) = pivot;
        if k < n
            L(k+1:n, k) = sparse(v(k+1:n) / pivot);
        end
        if show_steps
            fprintf('  Pivot U(k,k)=% .6g; nnz in new L column=%d\n', ...
                pivot, nnz(L(:, k)));
        end
    end
end

function contributors = ReachEarlierColumns(L, seeds, k)
% Predict the possible U(1:k-1,k) pattern by graph reachability.
% An edge j -> i means column j can change residual entry i. Since L is
% triangular, sorting reached indices gives a valid topological order.
    reached = false(k-1, 1);
    reached(seeds) = true;
    pending = seeds(:);
    while ~isempty(pending)
        j = pending(end);
        pending(end) = [];
        children = find(L(j+1:k-1, j)) + j;
        fresh = children(~reached(children));
        reached(fresh) = true;
        pending = [pending; fresh]; %#ok<AGROW>
    end
    contributors = find(reached);
% This convenient find() scans the marks, and each dense v costs O(n).
% Production sparse LU reuses workspaces and emits traversal order directly
% to avoid dimension-sized setup/scans for every column.
end
