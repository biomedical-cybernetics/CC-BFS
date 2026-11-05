function generate_ws_networks(Network_Sizes, Densities, Betas, num_realizations, result_folder, par_realizations, par_nets)
    if ~exist(result_folder, 'dir')
        mkdir(result_folder);
    end

    configs = generate_configs(Network_Sizes, Densities, Betas, result_folder, num_realizations);

    if par_nets && ~par_realizations
        start_parallel_pool(par_nets);
        process_parallel_nets(configs);
    elseif ~par_nets && par_realizations
        start_parallel_pool(par_realizations);
        process_parallel_realizations(configs);
    elseif par_nets && par_realizations
        error("Cannot parallelize both realization and networks.")
    else
        start_parallel_pool(0);
        process_sequential(configs);
    end
end

function configs = generate_configs(Network_Sizes, Densities, Betas, result_folder, num_realizations)
    configs = [];

    if exist('WattsStrogatz', 'file') ~= 2 %#ok<EXIST>
        error('Built-in function WattsStrogatz is not available on the MATLAB path.');
    end

    for N = reshape(Network_Sizes, 1, length(Network_Sizes))
        validateattributes(N, {'numeric'}, {'scalar','integer','>=',3,'finite'});
        for density = reshape(Densities, 1, length(Densities))
            validateattributes(density, {'numeric'}, {'scalar','>',0,'<=',1,'finite'});
            k = density_to_k(N, density);
            eff_density = (2 * k) / (N - 1);
            mean_degree = 2 * k;
            for beta = reshape(Betas, 1, length(Betas))
                validateattributes(beta, {'numeric'}, {'scalar','>=',0,'<=',1,'finite'});

                network_name = strcat('WS_with_N_', string(N), ...
                    '_density_', string(sprintf('%g', density)), ...
                    '_beta_', string(sprintf('%g', beta)));
                output = fullfile(result_folder, network_name + ".mat");

                if ~isfile(output)
                    fprintf('Adding to queue: %s\n', network_name);
                    configs = [configs; struct('N', N, 'density', density, 'beta', beta, ...
                        'k', k, 'mean_degree', mean_degree, 'eff_density', eff_density, 'network_name', network_name, ...
                        'output', output, 'num_realizations', num_realizations)]; %#ok<AGROW>
                else
                    fprintf('File %s already exists. Skipping...\n', output);
                end
            end
        end
    end
end

function process_sequential(configs)
    for i = 1:length(configs)
        config = configs(i);
        x_matrices = process_single_config(config, false);
        save(config.output, 'x_matrices', 'config', '-v7.3');
    end
end

function process_parallel_realizations(configs)
    for i = 1:length(configs)
        config = configs(i);
        x_matrices = process_single_config(config, true);
        save(config.output, 'x_matrices', 'config', '-v7.3');
    end
end

function process_parallel_nets(configs)
    pool = gcp('nocreate');
    if isempty(pool)
        num_threads = 1;
    else
        num_threads = pool.NumWorkers;
    end

    num_configs = length(configs);
    for batch_start = 1:num_threads:num_configs
        batch_end = min(batch_start + num_threads - 1, num_configs);
        batch_configs = configs(batch_start:batch_end);
        batch_size = length(batch_configs);

        fprintf('Processing batch %d-%d of %d networks...\n', batch_start, batch_end, num_configs);

        batch_x_matrices = cell(batch_size, 1);
        parfor i = 1:batch_size
            batch_x_matrices{i} = process_single_config(batch_configs(i), false);
        end

        fprintf('Saving batch %d-%d results...\n', batch_start, batch_end);
        save_batch_results(batch_configs, batch_x_matrices);
    end
end

function x_matrices = process_single_config(config, use_parallel)
    fprintf('Generating %s ... \n', config.network_name);
    num_realizations = config.num_realizations;

    x_matrices = cell(num_realizations, 1);
    if use_parallel
        parfor nidx = 1:num_realizations
            x_matrices{nidx} = generate_single_realization(config);
        end
    else
        for nidx = 1:num_realizations
            x_matrices{nidx} = generate_single_realization(config);
        end
    end
end

function x = generate_single_realization(config)
    G = WattsStrogatz(config.N, config.k, config.beta);
    x = logical(adjacency(G));
end

function save_batch_results(batch_configs, batch_x_matrices)
    for i = 1:length(batch_configs)
        config = batch_configs(i);
        x_matrices = batch_x_matrices{i};
        save(config.output, 'x_matrices', 'config', '-v7.3');
    end
end

function k = density_to_k(N, density)
    max_k = floor((N - 1) / 2);
    if max_k < 1
        error('N must be >= 3 for WattsStrogatz (requires k>=1).');
    end

    % In this WattsStrogatz implementation: mean degree = 2k, so density ~= 2k/(N-1).
    k = round((density * (N - 1)) / 2);
    k = max(1, min(k, max_k));
end

