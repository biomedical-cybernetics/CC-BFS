function generate_npso_networks(Network_Sizes, Densities, Ts, Gammas, Cs, num_realizations, result_folder, par_realizations, par_nets, save_geo)
    % Create result folder if it doesn't exist
    if ~exist(result_folder, 'dir')
        mkdir(result_folder);
    end

    % Generate configurations
    configs = generate_configs(Network_Sizes, Densities, Ts, Gammas, Cs, result_folder, save_geo, num_realizations);
    
    % Choose parallelization strategy
    if par_nets && ~par_realizations
        start_parallel_pool(par_nets);
        process_parallel_nets(configs);
    elseif ~par_nets && par_realizations
        start_parallel_pool(par_realizations);
        process_parallel_realizations(configs);
    elseif par_nets && par_realizations
        error("Cannot parallelize both realization and networks.")
    else
        % Default to sequential
        start_parallel_pool(0);
        process_sequential(configs);
    end
end

function configs = generate_configs(Network_Sizes, Densities, Ts, Gammas, Cs, result_folder, save_geo, num_realizations)
    configs = [];
    for N = reshape(Network_Sizes, 1, length(Network_Sizes))
        for T = reshape(Ts, 1, length(Ts))
            for density = reshape(Densities, 1, length(Densities))
                for gamma = reshape(Gammas,1 , length(Gammas))
                    for C = reshape(Cs,1 , length(Cs))
                        network_name = strcat('nPSO_with_N_', string(N), '_gamma_', string(gamma), ...
                            '_T_', string(T), '_density_', string(density), '_C_', string(C));
                        output = fullfile(result_folder, network_name + ".mat");

                        if ~isfile(output)
                            fprintf('Adding to queue: %s\n', network_name);
                            configs = [configs; struct('N', N, 'T', T, 'density', density, ...
                                'gamma', gamma, 'C', C, 'network_name', network_name, 'output', ...
                                output, 'save_geo', save_geo, 'num_realizations', num_realizations)]; %#ok<AGROW>
                        else
                            fprintf('File %s already exists. Skipping...\n', output);
                        end
                    end
                end
            end
        end
    end
end

function process_sequential(configs)
    % shutdown all parfor workers
    for i = 1:length(configs)
        [x_matrices, geo_matrices, conf] = process_single_config(configs(i), false);
        save(configs(i).output, 'x_matrices', 'conf', '-v7.3');
        if conf.save_geo
            save(configs(i).output, 'geo_matrices', '-append');
        end
    end
end

function process_parallel_realizations(configs)
    for i = 1:length(configs)
        [x_matrices, geo_matrices, conf] = process_single_config(configs(i), true);
        % Save immediately as we're not in parfor
        save(configs(i).output, 'x_matrices', 'conf', '-v7.3');
        if conf.save_geo
            save(configs(i).output, 'geo_matrices', '-append');
        end
    end
end

function process_parallel_nets(configs)
    % Get the number of threads from the current parallel pool
    pool = gcp('nocreate');
    if isempty(pool)
        num_threads = 1; % Default to 1 if no pool exists
    else
        num_threads = pool.NumWorkers;
    end
    
    num_configs = length(configs);
    
    % Process networks in batches of num_threads
    for batch_start = 1:num_threads:num_configs
        batch_end = min(batch_start + num_threads - 1, num_configs);
        batch_configs = configs(batch_start:batch_end);
        batch_size = length(batch_configs);
        
        fprintf('Processing batch %d-%d of %d networks...\n', batch_start, batch_end, num_configs);
        
        % Initialize storage for this batch
        batch_x_matrices = cell(batch_size, 1);
        batch_geo_matrices = cell(batch_size, 1);
        batch_configs_processed = cell(batch_size, 1);
        
        % Process current batch in parallel
        parfor i = 1:batch_size
            [x_matrices, geo_matrices, conf] = process_single_config(batch_configs(i), false);
            batch_x_matrices{i} = x_matrices;
            batch_geo_matrices{i} = geo_matrices;
            batch_configs_processed{i} = conf;
        end
        
        % Save results for this batch immediately
        fprintf('Saving batch %d-%d results...\n', batch_start, batch_end);
        save_batch_results(batch_configs, batch_x_matrices, batch_geo_matrices, batch_configs_processed);
    end
end

function [x_matrices, geo_matrices, conf] = process_single_config(config, use_parallel)
    fprintf('Generating %s ... \n', config.network_name);
    num_realizations = config.num_realizations;
    % Compute parameters
    x_matrices = cell(num_realizations, 1);
    geo_matrices = cell(num_realizations, 1);
    
    % Process realizations
    if use_parallel
        parfor nidx = 1:num_realizations
            [x_matrices{nidx}, geo_matrices{nidx}]  = generate_single_realization(config);
        end
    else
        for nidx = 1:num_realizations
            [x_matrices{nidx}, geo_matrices{nidx}]  = generate_single_realization(config);
        end
    end
    
    % Create configuration struct
    conf = struct('N', config.N, 'gamma', config.gamma, 'T', config.T, ...
        'C', config.C, 'density', config.density, ...
        'realizations', num_realizations, "save_geo", config.save_geo);
end

function [x, Geo] = generate_single_realization(config)
    plot_flag = 0;
    % compute half of node degree m
    m = compute_half_node_degree(config.N, config.density);
    if config.save_geo
        [x, ~, ~, Geo] = nPSO_model(config.N, m, config.T, config.gamma, config.C, plot_flag);
       
    else
        [x, ~, ~, ~] = nPSO_model(config.N, m, config.T, config.gamma, config.C, plot_flag);
        Geo = nan; % pseudo output
    end
    x = logical(x);
end

function save_batch_results(batch_configs, batch_x_matrices, batch_geo_matrices, batch_configs_processed)
% this function is used to save the results computed in parallel per batch
    for i = 1:length(batch_configs)
        x_matrices = batch_x_matrices{i};
        geo_matrices = batch_geo_matrices{i};
        conf = batch_configs_processed{i};
        save(batch_configs(i).output, 'x_matrices', 'conf', '-v7.3');
        if conf.save_geo
            save(batch_configs(i).output, 'geo_matrices', '-append');
        end
    end
end