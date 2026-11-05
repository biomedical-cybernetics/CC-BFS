function generate_test_networks()
% Quick smoke-test for network generation: reproduces one network from each
% benchmark configuration using n=1,000 nodes.
%
% Output: data/example_networks/test_generated/

script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
output_folder = fullfile(project_root, 'data', 'example_networks', 'test_generated');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

addpath(fullfile(script_dir, 'utils'));
addpath(fullfile(script_dir, 'utils', 'npso_model_generation'));
addpath(fullfile(script_dir, 'utils', 'ws_model_generation'));

fprintf('Output folder: %s\n\n', output_folder);

%% nPSO — varying density
fprintf('nPSO: varying density (n=1,000)...\n');
generate_npso_networks([1000], [0.001, 0.01, 0.1], [0.1], [2], [4], 1, output_folder, false, false, false);
fprintf('  done\n\n');

%% nPSO — varying temperature
fprintf('nPSO: varying temperature (n=1,000)...\n');
generate_npso_networks([1000], [0.05], [0.1, 0.5, 0.9], [2], [4], 1, output_folder, false, false, false);
fprintf('  done\n\n');

%% nPSO — varying gamma
fprintf('nPSO: varying gamma (n=1,000)...\n');
generate_npso_networks([1000], [0.05], [0.1], [2.0, 2.5, 3.0], [4], 1, output_folder, false, false, false);
fprintf('  done\n\n');

%% nPSO — varying communities
fprintf('nPSO: varying communities (n=1,000)...\n');
generate_npso_networks([1000], [0.1], [0.1], [2], [2, 4, 8], 1, output_folder, false, false, false);
fprintf('  done\n\n');

%% WS — varying beta, fixed density
fprintf('WS: varying beta, fixed density=0.1 (n=1,000)...\n');
generate_ws_networks([1000], 0.1, [0, 1e-3, 1e-2, 1e-1, 1], 1, output_folder, false, false);
fprintf('  done\n\n');

%% WS — varying density, fixed beta
fprintf('WS: varying density, fixed beta=0.05 (n=1,000)...\n');
generate_ws_networks([1000], [0.001, 0.01, 0.1], 0.05, 1, output_folder, false, false);
fprintf('  done\n\n');

mat_files = dir(fullfile(output_folder, '*.mat'));
fprintf('Done. %d networks written to %s\n', length(mat_files), output_folder);

end
