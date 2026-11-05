function benchmark_apsp_on_examples()
%BENCHMARK_APSP_ON_EXAMPLES Benchmark APSP algorithms on a folder of networks
%
% Benchmarks four APSP algorithms on every .mat file in the configured folder:
%   - ZM-BFS:  Standard BFS-based APSP
%   - CC-BFS:  The CC-BFS APSP algorithm
%   - CC-BFS-P: Shared-memory parallel CC-BFS
%   - DO-BFS:  Direction-Optimizing BFS baseline adapted from GAPBS
%
% Metrics recorded per algorithm per network:
%   - Running time (seconds)
%   - Theoretical dominant memory estimate (bytes)
%   - Empirical peak private memory delta (bytes)
%
% Configuration:
%   Edit DATA_FOLDER_RELATIVE below to point at any folder of .mat networks.
%   Use 'data/example_networks'   for the bundled example networks.
%   Use 'data/benchmark_dataset'  for the generated benchmark networks
%       (run generate_benchmark_artificial_networks.m first).
%
% Usage:
%   benchmark_apsp_on_examples()
%
% Requirements:
%   - MEX files must be compiled (call compile_mex_all with the Eigen path)
%   - Networks must contain adjacency matrix as variable 'x', 'adjMatrix',
%     or 'x_matrices'

%% ======================================================================
%% Configuration — edit DATA_FOLDER_RELATIVE to switch network folders
%% ======================================================================
DATA_FOLDER_RELATIVE = 'data/example_networks';
% DATA_FOLDER_RELATIVE = 'data/benchmark_dataset';  % generated benchmark networks
cc_bfs_p_threads = 4;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                  APSP ALGORITHM BENCHMARK                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Get script directory and set paths
script_dir = fileparts(mfilename('fullpath'));
mex_folder = fullfile(script_dir, '..', 'source', 'matlab', 'mex_functions');
data_folder = fullfile(script_dir, '..', DATA_FOLDER_RELATIVE);
result_folder = fullfile(script_dir, '..', 'results', 'comparison');

% Add MEX functions to path
addpath(mex_folder);

% Create result folder if it doesn't exist
if ~exist(result_folder, 'dir')
    mkdir(result_folder);
end

% Number of repetitions for timing
num_repetitions = 1;

% Algorithm configurations
algorithms = struct(...
    'name', {}, ...
    'func', {}, ...
    'params', {}, ...
    'type', {});

% ZM-BFS (BFS)
algorithms(end+1).name = 'ZM-BFS';
algorithms(end).func = 'bfs_apsp_l_cpp';
algorithms(end).params = '(x)';
algorithms(end).type = 'apsp';

% CC-BFS
algorithms(end+1).name = 'CC-BFS';
algorithms(end).func = 'cc_bfs_apsp_l_cpp';
algorithms(end).params = '(x)';
algorithms(end).type = 'apsp';

% CC-BFS-P
algorithms(end+1).name = sprintf('CC-BFS-P (%d threads)', cc_bfs_p_threads);
algorithms(end).func = 'cc_bfs_p';
algorithms(end).params = sprintf('(x, %d, ''ndr'')', cc_bfs_p_threads);
algorithms(end).type = 'apsp';

% DO-BFS
algorithms(end+1).name = 'DO-BFS';
algorithms(end).func = 'do_bfs_gapbs_apsp_l_cpp';
algorithms(end).params = '(x)';
algorithms(end).type = 'apsp';


%% ======================================================================
%% Load Networks
%% ======================================================================

fprintf('📂 Loading networks from: %s\n', data_folder);
fprintf('\n');

if ~exist(data_folder, 'dir')
    error('Example networks folder not found: %s\n', data_folder);
end

% Get all .mat files
files = dir(fullfile(data_folder, '*.mat'));

if isempty(files)
    error('No .mat files found in: %s\n', data_folder);
end

fprintf('   Found %d network(s)\n', length(files));
fprintf('\n');

%% ======================================================================
%% Run Comparisons
%% ======================================================================

% Storage for results - one row per network
results_table = cell(length(files), 4 + 3 * length(algorithms));
header = {'Network', 'Nodes', 'Edges', 'Density'};
for algo_idx = 1:length(algorithms)
    base_name = matlab.lang.makeValidName(algorithms(algo_idx).name);
    base_col = 4 + (algo_idx - 1) * 3;
    header{base_col + 1} = [base_name '_time_s'];
    header{base_col + 2} = [base_name '_theory_bytes'];
    header{base_col + 3} = [base_name '_process_bytes'];
end

for file_idx = 1:length(files)
    filepath = fullfile(data_folder, files(file_idx).name);
    [~, network_name, ~] = fileparts(filepath);
    
    % Print network header
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('Network %d/%d: %s\n', file_idx, length(files), network_name);
    fprintf('═══════════════════════════════════════════════════════════════\n');
    
    % Load adjacency matrix
    x = load_network(filepath);
    n = size(x, 1);
    m = nnz(x) / 2;  % Assuming undirected
    density = 2 * m / (n * (n - 1));
    
    % Network info
    fprintf('   Nodes:       %s\n', format_number(n));
    fprintf('   Edges:       %s\n', format_number(m));
    fprintf('   Density:     %.4f%%\n', density * 100);
    fprintf('   Repetitions: %d\n', num_repetitions);
    fprintf('\n');
    
    % Store network info in results table
    results_table{file_idx, 1} = network_name;
    results_table{file_idx, 2} = n;
    results_table{file_idx, 3} = m;
    results_table{file_idx, 4} = density;
    
    % Run each algorithm
    for algo_idx = 1:length(algorithms)
        algo = algorithms(algo_idx);
        
        fprintf('─────────────────────────────────────────────────────────────\n');
        fprintf('Algorithm: %s (%s)\n', algo.name, algo.func);
        fprintf('─────────────────────────────────────────────────────────────\n');
        
        % Run multiple times for timing
        times = zeros(num_repetitions, 1);
        theory_memories = NaN(num_repetitions, 1);
        process_memories = NaN(num_repetitions, 1);
        
        for rep = 1:num_repetitions
            fprintf('   Run %d/%d... ', rep, num_repetitions);
            
            try
                % Use evalc to suppress MEX function output
                % APSP algorithms return elapsed time as the second output
                % and counters as the third output.
                evalc(['[~, elapsed, counters] = ' algo.func algo.params ';']);
                times(rep) = elapsed;
                [theory_memories(rep), process_memories(rep)] = extract_memory_metrics(counters);
                fprintf('%.4f seconds\n', elapsed);
            catch ME
                fprintf('ERROR: %s\n', ME.message);
                times(rep) = NaN;
            end
        end
        
        % Compute statistics
        valid_times = times(~isnan(times));
        if isempty(valid_times)
            fprintf('   ❌ All runs failed!\n\n');
            continue;
        end
        
        time_mean = mean(valid_times);
        theory_bytes = mean(theory_memories(~isnan(theory_memories)));
        process_bytes = mean(process_memories(~isnan(process_memories)));
        
        % Print statistics
        fprintf('\n');
        fprintf('   Mean Time:        %.6f s\n', time_mean);
        fprintf('   Dominant Theoretical Memory: %s\n', format_bytes(theory_bytes));
        fprintf('   Empirical Memory: %s\n', format_bytes(process_bytes));
        fprintf('\n');
        
        % Store metrics in results table
        base_col = 4 + (algo_idx - 1) * 3;
        results_table{file_idx, base_col + 1} = time_mean;
        results_table{file_idx, base_col + 2} = theory_bytes;
        results_table{file_idx, base_col + 3} = process_bytes;
    end
    
    fprintf('\n');
end

%% ======================================================================
%% Summary
%% ======================================================================

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                         SUMMARY                                ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% Print summary table
fprintf('Performance Overview (Mean Time in seconds):\n');
fprintf('\n');

% Print header
fprintf('%-40s %8s %10s %10s', 'Network', 'Nodes', 'Edges', 'Density');
for algo_idx = 1:length(algorithms)
    fprintf(' %12s', algorithms(algo_idx).name);
end
fprintf('\n');
fprintf('%s\n', repmat('─', 1, 40 + 8 + 10 + 10 + 12 * length(algorithms) + length(algorithms)));

% Print data rows
for i = 1:size(results_table, 1)
    fprintf('%-40s %8d %10.0f %10.6f', ...
            results_table{i, 1}, results_table{i, 2}, results_table{i, 3}, results_table{i, 4});
    for algo_idx = 1:length(algorithms)
        base_col = 4 + (algo_idx - 1) * 3;
        time_val = results_table{i, base_col + 1};
        if ~isempty(time_val)
            fprintf(' %12.6f', time_val);
        else
            fprintf(' %12s', 'N/A');
        end
    end
    fprintf('\n');
end

fprintf('\n');

fprintf('Memory Usage:\n');
fprintf('\n');
fprintf('%-40s', 'Network');
for algo_idx = 1:length(algorithms)
    fprintf(' %22s %22s', [algorithms(algo_idx).name ' theory'], [algorithms(algo_idx).name ' empirical']);
end
fprintf('\n');
fprintf('%s\n', repmat('─', 1, 40 + 44 * length(algorithms) + 2 * length(algorithms)));

for i = 1:size(results_table, 1)
    fprintf('%-40s', results_table{i, 1});
    for algo_idx = 1:length(algorithms)
        base_col = 4 + (algo_idx - 1) * 3;
        theory_val = results_table{i, base_col + 2};
        process_val = results_table{i, base_col + 3};
        fprintf(' %22s %22s', format_bytes(theory_val), format_bytes(process_val));
    end
    fprintf('\n');
end

fprintf('\n');

%% ======================================================================
%% Save Results
%% ======================================================================

% Convert to table for CSV export
result_table = cell2table(results_table, 'VariableNames', header);

% Save as CSV
csv_file = fullfile(result_folder, 'comparison_results_matlab.csv');
writetable(result_table, csv_file);
fprintf('✅ Results saved to: %s\n', csv_file);
fprintf('\n');

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                   COMPARISON COMPLETE                          ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

end

%% ======================================================================
%% Helper Functions
%% ======================================================================

function x = load_network(filepath)
%LOAD_NETWORK Load adjacency matrix from .mat file
    fileInfo = whos('-file', filepath);
    varNames = {fileInfo.name};
    
    if ismember('x', varNames)
        data = load(filepath, 'x');
        x = data.x;
    elseif ismember('adjMatrix', varNames)
        data = load(filepath, 'adjMatrix');
        x = data.adjMatrix;
    elseif ismember('x_matrices', varNames)
        data = load(filepath, 'x_matrices');
        % Extract first element from cell array
        x = data.x_matrices{1};
    else
        error('No adjacency matrix found in file: %s\nExpected variable name: ''x'', ''adjMatrix'', or ''x_matrices''', filepath);
    end
    
end

function str = format_number(num)
%FORMAT_NUMBER Format number with thousands separators
    str = num2str(num);
    if num >= 1000
        % Add commas as thousands separators
        str = regexprep(str, '(\d{1,3})(?=(\d{3})+(?!\d))', '$1,');
    end
end

function [theory_bytes, process_bytes] = extract_memory_metrics(counters)
%EXTRACT_MEMORY_METRICS Read benchmark memory metrics from counters struct
    theory_bytes = NaN;
    process_bytes = NaN;

    if isstruct(counters)
        if isfield(counters, 'estimated_dominant_memory_bytes')
            theory_bytes = double(counters.estimated_dominant_memory_bytes);
        end
        if isfield(counters, 'process_delta_peak_private_bytes')
            process_bytes = double(counters.process_delta_peak_private_bytes);
        end
    end
end

function str = format_bytes(n_bytes)
%FORMAT_BYTES Format a byte count in a human-readable form
    if isempty(n_bytes) || ~isfinite(n_bytes) || n_bytes <= 0
        str = 'N/A';
    elseif n_bytes >= 2^30
        str = sprintf('%.2f GiB', n_bytes / 2^30);
    elseif n_bytes >= 2^20
        str = sprintf('%.2f MiB', n_bytes / 2^20);
    elseif n_bytes >= 2^10
        str = sprintf('%.2f KiB', n_bytes / 2^10);
    else
        str = sprintf('%.0f B', n_bytes);
    end
end

