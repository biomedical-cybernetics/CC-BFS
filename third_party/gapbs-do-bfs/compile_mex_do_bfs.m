this_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(this_dir, '..', '..');
matlab_dir = fullfile(repo_root, 'source', 'matlab');
addpath(matlab_dir);

eigen_include_dir = 'C:/path/to/eigen-3.4.0';
if strcmp(eigen_include_dir, 'C:/path/to/eigen-3.4.0')
    error('Set eigen_include_dir to your Eigen 3.4.0 install root (the directory that contains the Eigen folder).');
end

cpp_file = fullfile(this_dir, 'do_bfs_gapbs_apsp_l_cpp.cpp');
output_dir = fullfile(repo_root, 'source', 'matlab', 'mex_functions');

compile_mex_helper(cpp_file, output_dir, eigen_include_dir);
