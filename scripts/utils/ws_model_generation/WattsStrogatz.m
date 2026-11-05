
% Copyright 2015 The MathWorks, Inc.

function h = WattsStrogatz(N,K,beta)
% H = WattsStrogatz(N,K,beta) returns a Watts-Strogatz model graph with N
% nodes, N*K edges, mean node degree 2*K, and rewiring probability beta.
%
% beta = 0 is a ring lattice, and beta = 1 is a random graph.

% Connect each node to its K next and previous neighbors. This constructs
% indices for a ring lattice.
s = repelem((1:N)',1,K);
t = s + repmat(1:K,N,1);
t = mod(t-1,N)+1;

% Rewire the target node of each edge with probability beta
show_progress = true;
if exist('getCurrentTask', 'file') == 2 %#ok<EXIST>
    % In a parallel worker, Command Window output is forwarded and cannot be
    % updated in-place reliably.
    show_progress = isempty(getCurrentTask());
end

last_percent = -1;
last_msg_len = 0;
for source=1:N    
    switchEdge = rand(K, 1) < beta;
    
    newTargets = rand(N, 1);
    newTargets(source) = 0;
    newTargets(s(t==source)) = 0;
    newTargets(t(source, ~switchEdge)) = 0;
    
    [~, ind] = sort(newTargets, 'descend');
    t(source, switchEdge) = ind(1:nnz(switchEdge));

    percent = floor(100 * source / N);
    if show_progress && percent ~= last_percent
        msg = sprintf('WattsStrogatz progress: %3d%%', percent);
        if last_msg_len > 0
            fprintf(repmat('\b', 1, last_msg_len));
        end
        fprintf('%s', msg);
        last_msg_len = numel(msg);
        last_percent = percent;
        drawnow limitrate;
    end
end
if show_progress
    if last_msg_len > 0
        fprintf(repmat('\b', 1, last_msg_len));
    end
    fprintf('WattsStrogatz progress: 100%%\n');
end

h = graph(s,t);
end