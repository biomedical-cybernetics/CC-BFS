function compile_mex_helper(cpp_file_path, output_dir, eigen_include_dir)
    % COMPILE_MEX_HELPER - Compiles MEX files with common settings
    %
    % Usage: compile_mex_helper('path/to/file.cpp', 'output/dir/', 'path/to/eigen')
    %        compile_mex_helper('path/to/file.cpp', [], 'path/to/eigen')
    %
    % Default output_dir is source/matlab/mex_functions next to this helper.

    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile(fileparts(mfilename('fullpath')), 'mex_functions');
    end

    if nargin < 3 || isempty(eigen_include_dir)
        error('compile_mex_helper:MissingEigenIncludeDir', ...
            ['Eigen include directory is required. Pass the directory that contains ', ...
             'the Eigen folder, for example C:/path/to/eigen-3.4.0.']);
    end

    eigen_dense_header = fullfile(eigen_include_dir, 'Eigen', 'Dense');
    if ~isfolder(eigen_include_dir) || exist(eigen_dense_header, 'file') ~= 2
        error('compile_mex_helper:InvalidEigenIncludeDir', ...
            'Invalid Eigen include directory: %s\nExpected to find: %s', ...
            eigen_include_dir, eigen_dense_header);
    end

    % This helper uses MSVC-style flags and is intended for MATLAB + MSVC.
    mex('-largeArrayDims', ...
        'COMPFLAGS="$COMPFLAGS /std:c++17 /openmp /DMATLAB_BUILD"', ...
        'LDFLAGS="$LDFLAGS /openmp"', ...
        ['-I' eigen_include_dir], ...
        '-outdir', output_dir, ...
        cpp_file_path);

    fprintf('Successfully compiled: %s\n', cpp_file_path);
end
