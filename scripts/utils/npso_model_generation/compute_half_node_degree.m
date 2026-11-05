
function m =  compute_half_node_degree(N, density)
syms m;
% desired ratio of positive and negative samples after 10% link removal
assert((0<density) && (density<=1));
eqn = (m*(m+1)/2 + (N-m-1)*m) / (N*(N-1)/2) == density;
solutions = solve(eqn, m);
feasible_solutions = solutions(solutions < N);
if length(feasible_solutions) > 1
    error("more than one m are found");
end
m = max(round(double(feasible_solutions)), 1);

end



