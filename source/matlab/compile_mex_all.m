function compile_mex_all(eigen_include_dir)
%COMPILE_MEX_ALL Compile all released MATLAB MEX functions on Windows.

if nargin < 1 || isempty(eigen_include_dir)
    error('compile_mex_all:MissingEigenIncludeDir', ...
        'Pass the directory that directly contains the Eigen folder.');
end
if ~ispc
    error('compile_mex_all:UnsupportedPlatform', ...
        'The supplied MATLAB MEX build utilities are currently tested only on Windows.');
end

matlab_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(matlab_dir));
output_dir = fullfile(matlab_dir, 'mex_functions');
cpp_dir = fullfile(project_root, 'source', 'cpp');
third_party_dir = fullfile(project_root, 'third_party', 'gapbs-do-bfs');

compile_mex_helper(fullfile(cpp_dir, 'bfs_apsp_l_cpp.cpp'), output_dir, eigen_include_dir);
compile_mex_helper(fullfile(cpp_dir, 'cc_bfs_apsp_l_cpp.cpp'), output_dir, eigen_include_dir);
compile_mex_cc_bfs_p(output_dir, eigen_include_dir);
compile_mex_helper(fullfile(cpp_dir, 'cc_ess_cpp.cpp'), output_dir, eigen_include_dir);
compile_mex_helper(fullfile(cpp_dir, 'ess_cpp.cpp'), output_dir, eigen_include_dir);
compile_mex_helper(fullfile(third_party_dir, 'do_bfs_gapbs_apsp_l_cpp.cpp'), output_dir, eigen_include_dir);
end
