function start_parallel_pool(num_workers)
%START_PARALLEL_POOL Start or shut down a MATLAB parallel pool
%
%   start_parallel_pool(n)  starts a parallel pool with n workers (n > 0).
%                           If a pool with a different worker count already
%                           exists, it is replaced.
%   start_parallel_pool(0)  shuts down any existing pool (sequential mode).
%
%   If the Parallel Computing Toolbox is not available, this function
%   silently falls back to sequential execution.

    if nargin < 1
        num_workers = 0;
    end

    % --- Sequential mode: shut down any existing pool -------------------
    if num_workers <= 0
        if exist('gcp', 'file') == 2
            pool = gcp('nocreate');
            if ~isempty(pool)
                delete(pool);
                fprintf('Parallel pool shut down (sequential mode).\n');
            end
        end
        return;
    end

    % --- Parallel mode: start or reuse a pool ---------------------------
    try
        poolObj = gcp('nocreate');
        if isempty(poolObj)
            parpool(num_workers);
            fprintf('Started parallel pool with %d workers.\n', num_workers);
        elseif poolObj.NumWorkers ~= num_workers
            delete(poolObj);
            parpool(num_workers);
            fprintf('Restarted parallel pool with %d workers.\n', num_workers);
        else
            fprintf('Parallel pool with %d workers is already running.\n', num_workers);
        end
    catch ME
        warning('start_parallel_pool:NoParallel', ...
            'Could not start a parallel pool (%s). Running sequentially.', ME.message);
    end
end
