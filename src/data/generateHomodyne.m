% GENERATEHOMODYNE
%   Generate homodyne filtered phase from raw phase data.
%
%   Inputs
%   ------
%   INDIR       path to the directory containing `.mat` files that have raw
%               phase `phas`, magnitude `mag`, voxel size `vsz`, and
%               echo times `TEs`.  Subdirectories are allowed.
%   OUTDIR      path to the directory where the homodyne filtered phase will
%               be saved.
%   FILTERS     cell array of filters to apply
%   TE_CUTOFF   exclude individual echoes with TE < TE_CUTOFF. Such echoes will
%               still be used in multi-echo weighted averages/linear fits.
%
%   Outputs
%   -------
%   For each echo a `_e\d.mat` file is saved with the following variables:
%
%   `fl`        local field (reference / ground truth)
%   `hfl1`      homodyne filtered local field (step 1)
%   `hfl2`      processed homodyne filtered phase (step 2)
%   `mask`      binary brain mask
%   `vsz`       voxel size
%
%   For multi-echo data, an echotime weighted average (1), a magnitude and
%   echotime weighted average (2), and a linear fit (3) are also saved as
%   `_w\d.mat` files.
%
%   Notes
%   -----
%   Fermi windows, weighted averages/linear fits were not used in the paper.

INDIR = '../../data/raw';
OUTDIR = '../../data/homodyne';

% window, window size, window parameters
% negative window size: wsz = round(size(phas) ./ abs(wsz))
FILTERS = { ...
    'hann',     [64, 96, 128, -4], [], []; ...
    'hamming',  [64, 96, 128, -4], [], []; ...
    'gaussian', [64, 96, 128, -4], 4, []; ...
    'fermi',    -1, 8:2:14, [16, 32]; ...
    'unwrap',   -1, 4, [] ...
};

% exclude invdividual echoes with TE < TE_CUTOFF
TE_CUTOFF = 12e-3;

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
FILES = dir(fullfile(INDIR, '**/*.mat'));

% generateHomodyne
for ii = 1:length(FILES)
    fp = FILES(ii).folder;
    file = FILES(ii).name;
    [~, name, ~] = fileparts(file);

    outdir = replace(fp, INDIR, OUTDIR);
    if ~isfolder(outdir)
        mkdir(outdir);
    end

    fprintf('[%d/%d] Loading %s\n', ii, length(FILES), file);
    dat = load(fullfile(fp, file), 'mag', 'phas', 'vsz', 'TEs');
    MAG  = dat.mag;
    PHAS = dat.phas;
    vsz  = dat.vsz;
    TEs  = dat.TEs;
    nTE  = length(TEs);

    if nTE ~= size(PHAS, 4)
        error('Number of echoes does not match number of phase images')
    end

    % convert to s
    if TEs(1) > 1
        TEs = 1e-3 .* TEs;
    end

    % skip
    if nTE == 1 && TEs(1) < TE_CUTOFF
        continue
    end

    % start
    C = MAG .* exp(1i .* PHAS);

    fprintf('\tGenerating mask...\n')
    mask0 = generateMask(MAG(:,:,:,end), vsz, '-m -n -f 0.5');
    mask1 = erodeMask(mask0, 5);

    fprintf('\tUnwrapping phase...\n')
    UPHAS = unwrapLaplacian(PHAS, mask1, vsz);

    fprintf('\tRemoving background fields...\n')
    [FL, mask] = vsharp(UPHAS, mask1, vsz, 18*min(vsz):-2*max(vsz):2*max(vsz), 0.05);
    mask = erodeMask(mask, 1);
    [IX, IY, IZ] = cropIndices(mask);

    % add weighted average/linear fit for multi-echo data
    magfl = MAG;

    if nTE > 1
        mte = mean(TEs);
        TEs = reshape(TEs, 1, 1, 1, []);
        magw = sqrt(sum(MAG.^2, 4));

        % TE weighted
        w1 = mte .* sum(FL, 4) ./ sum(TEs, 4);

        % magnitude and TE weighted
        w2 = mte .* sum(MAG.*FL, 4) ./ sum(TEs.*MAG, 4);
        w2(~isfinite(w2)) = 0;

        % linear fit
        w3 = mte .* sum(MAG.^2.*TEs.*FL, 4) ./ sum(MAG.^2.*TEs.^2, 4);
        w3(~isfinite(w3)) = 0;

        FL(:,:,:,end+1) = w1;
        magfl(:,:,:,end+1) = magw;
        TEs(end+1) = mte;

        FL(:,:,:,end+1) = w2;
        magfl(:,:,:,end+1) = magw;
        TEs(end+1) = mte;

        FL(:,:,:,end+1) = w3;
        magfl(:,:,:,end+1) = magw;
        TEs(end+1) = mte;
    end

    Cfl = magfl .* exp(1i .* FL);
    SCL = B0 .* GAMMA .* TEs;

    % apply filters
    for ff = 1:size(FILTERS, 1)
        wname = FILTERS{ff, 1};
        wszs  = FILTERS{ff, 2};
        args1 = FILTERS{ff, 3};
        args2 = FILTERS{ff, 4};
        if isempty(args1), args1 = NaN; end
        if isempty(args2), args2 = NaN; end

        for ww = 1:length(wszs)
            wsz = wszs(ww);
            if wsz < 0
                wsz = round(size(PHAS) ./ abs(wsz));
            end

            for aa = 1:length(args1)
                for bb = 1:length(args2)
                    % filtering unwrapped phase
                    if strcmpi(wname, 'unwrap')
                        sigma = args1(aa);
                        ID = [wname, '_s_', num2str(sigma)];
                        fprintf('\tFiltering: %s, sigma %f...\n', wname, sigma);
                        % filter local fields
                        HFL1 = FL - imgaussfilt(FL, sigma);
                        % filter unwrapped phase
                        UHPHAS = UPHAS - imgaussfilt(UPHAS, sigma);

                    % homodyne Fermi filter
                    elseif strcmpi(wname, 'fermi')
                        wf = args1(aa);
                        rf = args2(bb);
                        ID = [wname, '_n_', num2str(wszs(ww)), '_w_', num2str(wf), '_r_', num2str(rf)];

                        fprintf('\tFiltering: %s, n %d, wf %f, rf %f...\n', wname, wsz(1), wf(1), rf(1));
                        % filter local fields
                        HFL1 = angle(homodyne(Cfl, wsz, wname, wf, rf));
                        % filter raw phase
                        HPHAS = angle(homodyne(C, wsz, wname, wf, rf));
                        fprintf('\tUnwrapping filtered phase...\n')
                        UHPHAS = unwrapLaplacian(HPHAS, mask1, vsz);

                    % homodyne Gaussian filter
                    elseif strcmpi(wname, 'gaussian')
                        sigma = 1 / args1(aa);
                        ID = [wname, '_n_', num2str(wszs(ww)), '_s_', num2str(sigma)];

                        fprintf('\tFiltering: %s, n %d, sigma %f...\n', wname, wsz(1), sigma);
                        % filter local fields
                        HFL1 = angle(homodyne(Cfl, wsz, wname, sigma));
                        % filter raw phase
                        HPHAS = angle(homodyne(C, wsz, wname, sigma));
                        fprintf('\tUnwrapping filtered phase...\n')
                        UHPHAS = unwrapLaplacian(HPHAS, mask1, vsz);

                    % homodyne Hann, Hamming filters
                    else
                        ID = [wname, '_n_', num2str(wszs(ww))];
                        fprintf('\tFiltering: %s, n %d...\n', wname, wsz(1));
                        % filter local fields
                        HFL1 = angle(homodyne(Cfl, wsz, wname));
                        % filter raw phase
                        HPHAS = angle(homodyne(C, wsz, wname));
                        fprintf('\tUnwrapping filtered phase...\n')
                        UHPHAS = unwrapLaplacian(HPHAS, mask1, vsz);
                    end

                    % weighted average/linear fit for multi-echo data
                    if nTE > 1
                        TEs = TEs(1:size(UHPHAS, 4));
                        % TE weighted
                        w1 = mte .* sum(UHPHAS, 4) ./ sum(TEs, 4);

                        % magnitude and TE weighted
                        w2 = mte .* sum(MAG.*UHPHAS, 4) ./ sum(TEs.*MAG, 4);
                        w2(~isfinite(w2)) = 0;

                        % linear fit
                        w3 = mte .* sum(MAG.^2.*TEs.*UHPHAS, 4) ./ sum(MAG.^2.*TEs.^2, 4);
                        w3(~isfinite(w3)) = 0;

                        UHPHAS(:,:,:,end+1) = w1;
                        TEs(end+1) = 1;

                        UHPHAS(:,:,:,end+1) = w2;
                        TEs(end+1) = 1;

                        UHPHAS(:,:,:,end+1) = w3;
                        TEs(end+1) = 1;
                    end

                    fprintf('\t\tRemoving Background fields...\n')
                    HFL2 = vsharp(UHPHAS, mask1, vsz, 18*min(vsz):-2*max(vsz):2*max(vsz), 0.05);

                    % save
                    outpath = fullfile(outdir, [name, '_', ID]);

                    for te = 1:size(FL, 4)
                        if nTE == 1
                            outpath1 = [outpath, '.mat'];
                        elseif te > nTE
                            outpath1 = [outpath, '_w', num2str(te-nTE), '.mat'];
                        else
                            if TEs(te) < TE_CUTOFF
                                continue
                            end
                            outpath1 = [outpath, '_e', num2str(te), '.mat'];
                        end

                        % convert units
                        scl  = SCL(te);
                        fl   = mask .* single(FL(:,:,:,te) ./ scl);
                        hfl1 = mask .* single(HFL1(:,:,:,te) ./ scl);
                        hfl2 = mask .* single(HFL2(:,:,:,te) ./ scl);

                        fprintf('\t\tSaving %s\n', outpath1);
                        save(outpath1, 'fl', 'hfl1', 'hfl2', 'mask', 'vsz', '-v7.3')
                    end
                    fprintf('\n')
                end
            end
        end
    end
    fprintf('\n')
end
