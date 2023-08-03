% PREPROCESS
%   Pre-processes SWI filtered phase data.
%
%   Inputs
%   ------
%   INDIR       path to the directory containing SWI filtered phase data. The
%               directory can include subdirectories. See `GLOB`.
%   OUTDIR      path to the directory where preprocessed SWI phase data will be
%               saved.
%   GLOB        glob pattern used to match files in `INDIR`.
%               The default pattern is '*.mat'. MAT and NIfTI file formats are
%               supported. If SWI files are saved as NIfTI, uncomment the
%               required lines below and set `MAGFILE` to locate the
%               corresponding magnitude file.
%   IS_GE       GE phase is scaled to [-1000 pi, 1000 pi]. If true, phase is
%               scaled accordingly: `phas = phas ./ 1000`
%
%   Output
%   ------
%   For each SWI filtered phase a `_fl.mat` file is saved with the following
%   variables:
%
%   `fl`        local field, i.e. processed SWI phase

INDIR = '../../data/swi';
OUTDIR = '../../data/swi';

% mat files containing: mag, phas, vsz, TE
GLOB = '*.mat';

% nifti phase/magnitude files
%GLOB = '*phase*.nii*';
%MAGFILE = @(file) replace(file, 'phase', 'mag');

% GE has different scaling factors for phase
IS_GE = false;

% if TE is not provided (Philips)
DEFAULT_TE = 0.02;  % in seconds

% constants
B0 = 3;             % field strength (T)
GAMMA = 267.522;    % gyromagnetic ratio (10^6 rad/(s*T))

% create output directory if it doesn't exist
if ~isfolder(OUTDIR)
    mkdir(OUTDIR);
end

% convert relative path to absolute path
[~, info] = fileattrib(INDIR);
INDIR = info.Name;

[~, info] = fileattrib(OUTDIR);
OUTDIR = info.Name;

% get files to process
FILES = dir(fullfile(INDIR, '**', GLOB));

% preprocess
for ii = 1:length(FILES)
    fp = FILES(ii).folder;
    file = FILES(ii).name;
    [~, name, ~] = fileparts(file);

    outdir = replace(fp, INDIR, OUTDIR);
    if ~isfolder(outdir)
        mkdir(outdir);
    end

    fprintf('[%d/%d] Loading %s\n', ii, length(FILES), file);
    if contains(file, '.mat')
        dat = load(fullfile(fp, file), 'mag', 'phas', 'vsz', 'TE');
        phas = dat.phas;    % homodyne filtered phase
        mag  = dat.mag;     % magnitude for mask generation
        vsz  = dat.vsz;     % voxel size in mm
        TE   = dat.TE;      % in seconds

    elseif contains(file, '.nii')
        niip = load_untouch_nii(fullfile(fp, file));
        niim = load_untouch_nii(fullfile(fp, MAGFILE(file)));

        phas = single(niip.img);
        mag  = single(niim.img);
        vsz  = niip.hdr.dime.pixdim(2:4);

        % get TE from nifti header (dcm2niix)
        try
            TE = regexpi(niip.hdr.hist.descrip, 'TE=\d*', 'match');
            TE = TE{1};
            TE = str2double(TE(4:end));
        catch
            fprintf('\t[WARN] Could not find TE in nifti header. Using 20 ms.\n')
            TE = DEFAULT_TE;
        end
    end

    if TE == 0
        TE = DEFAULT_TE;
    end

    % convert TE to seconds
    if TE > 1
        TE = 1e-3 * TE;
    end

    % scale phase
    if max(vec(phas)) > 3.142 || min(vec(phas)) < -3.142
        if IS_GE
            phas = phas ./ 1000;
        else
            phas = rescale(phas, -pi, pi);
        end
    end

    fprintf('\tGenerating mask...\n')
    mask0 = generateMask(mag, vsz, '-m -n -f 0.5');
    mask1 = erodeMask(mask0, 5);

    fprintf('\tUnwrapping phase...\n')
    uphas = unwrapLaplacian(phas, mask1, vsz);

    fprintf('\tRemoving background fields...\n')
    [fl, mask] = vsharp(uphas, mask1, vsz, 18*min(vsz):-2*max(vsz):2*max(vsz), 0.05);
    mask = erodeMask(mask, 1);

    % convert units
    scl = B0 .* GAMMA .* TE;
    fl  = mask .* single(fl ./ scl);

    outpath1 = fullfile(outdir, [name, '_fl.mat']);
    fprintf('\tSaving... %s\n', outpath1);
    save(outpath1, 'fl', '-v7.3')

    fprintf('\n')
end
