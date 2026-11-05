function compile_mex_cc_bfs_p(output_dir, eigen_include_dir)
%COMPILE_MEX_CC_BFS_P Compile CC-BFS-P using MATLAB's Intel OpenMP runtime.

if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'mex_functions');
end
if nargin < 2 || isempty(eigen_include_dir)
    error('compile_mex_cc_bfs_p:MissingEigenIncludeDir', ...
        'Pass the directory that directly contains Eigen.');
end
if ~ispc
    error('compile_mex_cc_bfs_p:UnsupportedPlatform', ...
        'The supplied CC-BFS-P MEX build command is currently tested only on Windows.');
end

matlab_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(matlab_dir));
source_dir = fullfile(project_root, 'source', 'cpp', 'cc_bfs_p');
entry_file = fullfile(source_dir, 'cc_bfs_p.cpp');
matlab_lib_dir = fullfile(matlabroot, 'bin', 'win64');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

mex('-largeArrayDims', ...
    'COMPFLAGS="$COMPFLAGS /std:c++17 /openmp /DMATLAB_BUILD"', ...
    'LDFLAGS="$LDFLAGS /openmp"', ...
    ['LINKFLAGS="$LINKFLAGS /nodefaultlib:vcomp /LIBPATH:"' ...
        matlab_lib_dir '" libiomp5md.lib"'], ...
    ['-I' eigen_include_dir], ...
    ['-I' fullfile(source_dir, 'include')], ...
    '-outdir', output_dir, ...
    '-output', 'cc_bfs_p', ...
    entry_file);

fprintf('Successfully compiled CC-BFS-P: %s\n', entry_file);
end
