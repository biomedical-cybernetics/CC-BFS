function generate_benchmark_artificial_networks()
%GENERATE_BENCHMARK_ARTIFICIAL_NETWORKS Generate artificial benchmark networks.
%   Generates the active nPSO and Watts-Strogatz network grids used by the
%   working repository benchmarks. RGG networks are intentionally not included.

    fprintf('Artificial Network Generation for CC-BFS Benchmarking\n\n');

    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);
    output_folder = fullfile(project_root, 'data', 'benchmark_dataset');
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    addpath(fullfile(script_dir, 'utils'));
    addpath(fullfile(script_dir, 'utils', 'npso_model_generation'));
    addpath(fullfile(script_dir, 'utils', 'ws_model_generation'));

    fprintf('Output folder: %s\n\n', output_folder);

    realizations = 1;
    save_geo = false;
    par_realizations = false;
    par_nets = false;

    %% nPSO networks: active Exp. 01/02 benchmark grid.
    fprintf('nPSO: varying gamma for N=100000.\n');
    generate_npso_networks(100000, 0.1, 0.1, [2, 2.5, 3], 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);

    fprintf('nPSO: varying temperature for N=100000.\n');
    generate_npso_networks(100000, 0.1, [0.1, 0.5, 0.9], 2, 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);

    fprintf('nPSO: varying density for N=100000.\n');
    generate_npso_networks(100000, ...
        [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.9], ...
        0.1, 2, 4, realizations, output_folder, par_realizations, par_nets, save_geo);

    fprintf('nPSO: varying community count for N=100000.\n');
    generate_npso_networks(100000, 0.1, 0.1, 2, [2, 4, 6, 8, 10], ...
        realizations, output_folder, par_realizations, par_nets, save_geo);

    fprintf('nPSO: smaller N/density grid.\n');
    generate_npso_networks([100, 1000, 10000], ...
        [0.01, 0.025, 0.05, 0.1, 0.25, 0.5], ...
        0.1, 2, 4, realizations, output_folder, par_realizations, par_nets, save_geo);

    fprintf('nPSO: size/gamma support grid.\n');
    generate_npso_networks([1000, 10000], 0.1, 0.1, [2, 3], 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);
    generate_npso_networks(100000, 0.1, 0.1, [2, 3], 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);

    fprintf('nPSO: size/density support grid.\n');
    generate_npso_networks([1000, 10000], [0.01, 0.1], 0.1, 2, 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);
    generate_npso_networks(100000, [0.01, 0.1], 0.1, 2, 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);

    fprintf('nPSO: size/temperature support grid.\n');
    generate_npso_networks([1000, 10000], 0.1, [0.1, 0.5], 2, 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);
    generate_npso_networks(100000, 0.1, [0.1, 0.5], 2, 4, ...
        realizations, output_folder, par_realizations, par_nets, save_geo);

    %% Watts-Strogatz networks: active 10k and 100k benchmark grids.
    fprintf('Watts-Strogatz: 10k density/beta grid.\n');
    generate_ws_networks(10000, ...
        [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], ...
        [0, 1e-3, 1e-2, 1e-1, 1], ...
        realizations, output_folder, par_realizations, par_nets);

    fprintf('Watts-Strogatz: 100k density grid at beta=0.05.\n');
    generate_ws_networks(100000, ...
        [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1], ...
        0.05, realizations, output_folder, par_realizations, par_nets);

    fprintf('Watts-Strogatz: 100k beta grid at density=0.1.\n');
    generate_ws_networks(100000, 0.1, [0, 1e-3, 1e-2, 1e-1, 1], ...
        realizations, output_folder, par_realizations, par_nets);

    mat_files = dir(fullfile(output_folder, '*.mat'));
    fprintf('\nNetwork generation complete. Output folder: %s\n', output_folder);
    fprintf('Current .mat files in output folder: %d\n', numel(mat_files));
end
