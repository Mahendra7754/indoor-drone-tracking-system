%% KF analysis: dropout-focused thesis figures (MATLAB R2020a or newer)
% Run this script with the CSV alongside it, or select the CSV when prompted.
% Output: three separate XYZ position/difference PNGs at 600 dpi and CSV tables.
% Colours match stationary_triangle_precision_reference_style5.m.
%
% Interpretation:
% - 'Stereo measurement' is the pre-KF position. In the supplied tracking
%   pipeline, axis scaling has ALREADY been applied. Do not apply it again.
% - Stereo minus KF is an estimator difference, not ground-truth error.
% - Differentiated position is a noisy comparison, not true velocity.
% - Logged sigma is model-based sqrt(P_ii), not measured positioning error.
% - Interval SD measures repeatability only if the marker was stationary.
% Dropout analysis:
% - Pale orange bands D1, D2, ... mark logged prediction-only runs.
% - Band boundaries use adjacent timestamp midpoints, not exact event times.
% - Long logging gaps split runs and are never shaded as known dropouts.
% - Orange diamonds mark the FIRST LOGGED measurement following a run.
% - Recovery differences are POST-update, not pre-update innovations.
% - The supplied logger records every >=0.1 s, less often than KF execution.
%   Unlogged updates/short dropouts can occur. Exact innovations and physical
%   dropout durations cannot be recovered from this CSV. No reconstruction
%   of a pre-update innovation is attempted.
% - Dropout RMS accuracy cannot be computed without an independent reference.
% All finite valid samples are retained. No smoothing/outlier filter is added.

clear; clc; close all;

%% USER SETTINGS
filename = 'triangle_tracking_tof_20260810_142052.csv';
startTime_s = NaN;                % Seconds from recording start, NaN = all
endTime_s = NaN;
precisionInterval_s = [0 5];       % Select a verified stationary interval
stationaryIntervalConfirmed = false; % Set true ONLY after checking experiment
maximumGap_s = NaN;               % NaN = 3 times median logging interval
markerTarget = 45;                % Sparse markers only, all line data retained
exportImages = true;
createAdditionalFigures = false;  % Optional velocity and filter diagnostics
resolutionDPI = 600;
showDropoutLabels = true;         % D1, D2, ... match dropout_summary.csv

scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
outputFolder = fullfile(scriptFolder, 'KF_report_figures');

%% APPROVED STYLE
C.blue = [80 145 190]/255;
C.orange = [222 82 24]/255;
C.teal = [90 163 141]/255;
C.red = [190 64 64]/255;
C.grey = [125 125 125]/255;
C.dark = [65 65 65]/255;
axisColours = [C.blue; C.orange; C.teal];
axisStyles = {'-', '--', ':'};
axisMarkers = {'o', 'x', '.'};
axisNames = {'X','Y','Z'};

%% INPUT AND VALIDITY
if ~isfile(filename)
    besideScript = fullfile(scriptFolder, filename);
    if isfile(besideScript)
        filename = besideScript;
    else
        [fileName, fileFolder] = uigetfile('*.csv', 'Select tracking CSV');
        if isequal(fileName,0), error('No measurement file selected.'); end
        filename = fullfile(fileFolder,fileName);
    end
end
T = readtable(filename);
required = {'time_s','triangle_valid', ...
    'rear_center_X_m','rear_center_Y_m','rear_center_Z_m', ...
    'kf_X_m','kf_Y_m','kf_Z_m','kf_Vx_mps','kf_Vy_mps','kf_Vz_mps'};
missing = setdiff(required,T.Properties.VariableNames);
assert(isempty(missing),'Missing columns: %s',strjoin(missing,', '));
assert(height(T)>=2,'At least two logged samples are required.');
t0 = T.time_s;
assert(all(isfinite(t0)) && all(diff(t0)>0), ...
    'Timestamps must be finite and strictly increasing. Check duplicate/reset times.');
t0 = t0-t0(1);
if isnan(maximumGap_s), maximumGap_s = 3*median(diff(t0)); end
assert(isfinite(maximumGap_s) && maximumGap_s>0,'maximumGap_s must be positive.');
selected = true(height(T),1);
if isfinite(startTime_s), selected = selected & t0>=startTime_s; end
if isfinite(endTime_s), selected = selected & t0<=endTime_s; end
T = T(selected,:); t = t0(selected);
assert(numel(t)>=3,'Selected interval must contain at least three samples.');
raw = 100*[T.rear_center_X_m,T.rear_center_Y_m,T.rear_center_Z_m];
kf = 100*[T.kf_X_m,T.kf_Y_m,T.kf_Z_m];
vKF = [T.kf_Vx_mps,T.kf_Vy_mps,T.kf_Vz_mps];
valid = T.triangle_valid==1 & all(isfinite(raw),2);
initialized = all(isfinite(kf),2);
if ismember('kf_initialized',T.Properties.VariableNames)
    initialized = initialized & T.kf_initialized==1;
end
raw(~valid,:) = NaN;
kf(~initialized,:) = NaN; vKF(~initialized,:) = NaN;
paired = valid & initialized;
assert(any(paired),'No paired valid stereo and KF position samples.');
if ismember('kf_prediction_only',T.Properties.VariableNames)
    prediction = initialized & T.kf_prediction_only==1;
    predictionLabel = 'KF prediction only';
else
    prediction = initialized & ~valid;
    predictionLabel = 'KF without stereo (inferred)';
end
sigma = nan(height(T),3);
sigmaColumns = {'kf_sigma_X_m','kf_sigma_Y_m','kf_sigma_Z_m'};
if all(ismember(sigmaColumns,T.Properties.VariableNames))
    sigma = 100*T{:,sigmaColumns};
    sigma(~initialized,:) = NaN;
    sigma(sigma<0) = NaN;
end

%% DROPOUT EVENTS FROM LOGGED SAMPLES
% Prefer the explicit KF flag. A missing valid marker after initialization
% is the fallback when the flag was not logged. Conflicting rows are excluded.
dropoutMask = prediction & ~valid;
[dropoutSummary,dropoutBands,recoveryIndices] = dropoutEvents( ...
    t,dropoutMask,paired,raw,kf,maximumGap_s);
fprintf('\n===== LOGGED DROPOUT RUNS =====\n');
disp(dropoutSummary);
fprintf(['%d logged runs, %d with a following logged measurement.\n' ...
    'Bands/durations are sample-based estimates. CSV logging may omit KF updates.\n' ...
    'Recovery differences below are POST-update, not innovations or dropout accuracy.\n'], ...
    height(dropoutSummary),numel(recoveryIndices));

%% NUMERICAL COMPARISONS
% Differentiate separate continuous valid runs. Never bridge a dropout.
% First/last points in each run use gradient's one-sided differences.
vDiff = nan(height(T),3);
indices = find(valid);
if ~isempty(indices)
    breaks = [1; find(diff(indices)>1 | diff(t(indices))>maximumGap_s)+1; numel(indices)+1];
    for r = 1:numel(breaks)-1
        ii = indices(breaks(r):breaks(r+1)-1);
        if numel(ii)>=3
            for j = 1:3
                vDiff(ii,j) = gradient(raw(ii,j)/100,t(ii));
            end
        end
    end
end
positionDifference = raw-kf;
velocityDifference = vDiff-vKF;
speedDiff = sqrt(sum(vDiff.^2,2));
speedKF = sqrt(sum(vKF.^2,2));
positionStats = differenceStats(positionDifference,axisNames);
velocityStats = differenceStats(velocityDifference,axisNames);
precisionMask = paired & t>=precisionInterval_s(1) & t<=precisionInterval_s(2);
sdRaw = nan(1,3); sdKF = nan(1,3);
if nnz(precisionMask)>=2
    sdRaw = std(raw(precisionMask,:),0,1);
    sdKF = std(kf(precisionMask,:),0,1);
end
precisionStats = table(string(axisNames(:)),repmat(nnz(precisionMask),3,1), ...
    sdRaw',sdKF','VariableNames',{'Axis','PairedN','StereoSD_cm','KFSD_cm'});
fprintf('\nPosition differences: stereo minus KF [cm], not accuracy errors\n');
disp(positionStats);
fprintf('\nVelocity differences: differentiated stereo minus KF [m/s]\n');
disp(velocityStats);
fprintf('\nSelected interval %.2f to %.2f s: SD [cm]\n',precisionInterval_s);
disp(precisionStats);
if ~stationaryIntervalConfirmed
    fprintf('Stationarity is unverified. Interval SD may include motion and filter settling.\n');
end
fprintf('Valid stereo: %d/%d samples. Prediction only: %d samples.\n', ...
    nnz(valid),numel(t),nnz(prediction));
fprintf('Gap threshold: %.4f s. No ToF accuracy comparison is made here.\n',maximumGap_s);

%% THREE AXIS FIGURES: POSITION ABOVE ITS STEREO-KF DIFFERENCE
positionFigures = cell(1,3);
positionNames = cell(1,3);
for j = 1:3
    positionFigures{j} = reportFigure( ...
        sprintf('%s position and difference',axisNames{j}),17,13);
    positionNames{j} = sprintf('KF_%s_position_and_difference.png',axisNames{j});
    L1 = tiledlayout(positionFigures{j},2,1,'TileSpacing','compact','Padding','compact');
    axPosition = nexttile(L1);
    [hRaw,hKF,hPred] = comparisonPlot(axPosition,t,raw(:,j),kf(:,j),prediction, ...
        C,maximumGap_s,markerTarget);
    ylabel(sprintf('%s [cm]',axisNames{j}));
    title(sprintf('(a) %s position',axisNames{j}));
    comparisonLegend(axPosition,hRaw,hKF,hPred,'Stereo measurement',predictionLabel);
    formatAxes(axPosition,C); xlim([t(1) t(end)]);
    markDropouts(axPosition,dropoutBands,C,showDropoutLabels);
    if ~isempty(recoveryIndices)
        plot(axPosition,t(recoveryIndices),raw(recoveryIndices,j),'d', ...
            'Color',C.orange,'MarkerFaceColor','w','MarkerSize',5, ...
            'LineWidth',1,'HandleVisibility','off');
    end
    title(axPosition,{sprintf('(a) %s position',axisNames{j}), ...
        'Shaded: logged dropout | diamond: first logged return'}, ...
        'FontSize',9,'FontWeight','normal');

    axDifference = nexttile(L1);
    hold(axDifference,'on');
    hDifference = drawSeries(axDifference,t,positionDifference(:,j),C.red,'-','.', ...
        maximumGap_s,markerTarget,0.85);
    yline(0,':','Color',C.grey,'HandleVisibility','off');
    ylabel(sprintf('\Delta%s [cm]',axisNames{j}));
    title(sprintf('(b) %s difference: stereo - KF (post-update)',axisNames{j}));
    if ~isempty(recoveryIndices)
        hRecovery = plot(axDifference,t(recoveryIndices),positionDifference(recoveryIndices,j), ...
            'd','Color',C.orange,'MarkerFaceColor','w','MarkerSize',5,'LineWidth',1);
        legend(axDifference,[hDifference hRecovery], ...
            {'Stereo - KF','First logged return'},'Location','best','FontSize',7);
    end
    text(0.03,0.04,sprintf('Valid-sample RMS difference = %.3f cm',positionStats.RMSDifference(j)), ...
        'Units','normalized','VerticalAlignment','bottom','FontSize',8,'BackgroundColor','w');
    formatAxes(axDifference,C); xlim([t(1) t(end)]);
    markDropouts(axDifference,dropoutBands,C,false);
    xlabel('Time [s]');
    linkaxes([axPosition axDifference],'x');
end

if createAdditionalFigures
%% FIGURE 2: VELOCITY COMPONENTS AND SPEED (2 x 2)
f2 = reportFigure('Velocity and speed comparison',19,15);
L2 = tiledlayout(f2,2,2,'TileSpacing','compact','Padding','compact');
a2 = gobjects(4,1);
for j = 1:4
    a2(j) = nexttile(L2);
    if j<=3
        measured = vDiff(:,j); estimated = vKF(:,j);
        label = sprintf('V_%s [m/s]',lower(axisNames{j}));
        panelTitle = sprintf('(%c) %s velocity','a'+j-1,axisNames{j});
    else
        measured = speedDiff; estimated = speedKF;
        label = 'Speed [m/s]'; panelTitle = '(d) Total speed';
    end
    [hRaw,hKF,hPred] = comparisonPlot(gca,t,measured,estimated,prediction,C,maximumGap_s,markerTarget);
    ylabel(label); title(panelTitle); xlabel('Time [s]');
    if j==1, comparisonLegend(gca,hRaw,hKF,hPred,'Differentiated stereo',predictionLabel); end
    formatAxes(gca,C); xlim([t(1) t(end)]);
end
linkaxes(a2,'x');

%% FIGURE 3: TRAJECTORY, MODEL UNCERTAINTY, INTERVAL SD AND AVAILABILITY
f3 = reportFigure('Filter diagnostics',19,16);
L3 = tiledlayout(f3,2,2,'TileSpacing','compact','Padding','compact');
ax = nexttile(L3); hold(ax,'on');
h1 = plot3(ax,raw(:,1),raw(:,2),raw(:,3),'.','Color',C.blue,'MarkerSize',3);
[~,trajectory] = insertGaps(t,kf,maximumGap_s);
h2 = plot3(ax,trajectory(:,1),trajectory(:,2),trajectory(:,3),'-','Color',C.teal,'LineWidth',1);
if any(prediction)
    plot3(ax,kf(prediction,1),kf(prediction,2),kf(prediction,3),'x', ...
        'Color',C.orange,'MarkerSize',3,'HandleVisibility','off');
end
xlabel('X [cm]'); ylabel('Y [cm]'); zlabel('Z [cm]');
title('(a) Rail trajectory, camera frame'); axis equal; view(35,24);
legend([h1 h2],{'Stereo measurement','Kalman filter'},'Location','best','FontSize',7);
formatAxes(ax,C);

ax = nexttile(L3); hold(ax,'on');
if any(isfinite(sigma(:)))
    hs = gobjects(3,1);
    for j = 1:3
        hs(j) = drawSeries(ax,t,sigma(:,j),axisColours(j,:),axisStyles{j}, ...
            axisMarkers{j},maximumGap_s,markerTarget,1);
    end
    legend(hs,{'\sigma_X','\sigma_Y','\sigma_Z'},'Location','best');
else
    text(ax,0.5,0.5,'No logged covariance data','Units','normalized','HorizontalAlignment','center');
end
title('(b) KF model-based uncertainty'); xlabel('Time [s]'); ylabel('1\sigma [cm]');
formatAxes(ax,C); xlim([t(1) t(end)]);

ax = nexttile(L3);
if nnz(precisionMask)>=2
    b = bar(ax,1:3,[sdRaw;sdKF]','grouped','BarWidth',0.72);
    b(1).FaceColor = C.blue; b(1).EdgeColor = C.blue;
    b(2).FaceColor = 'none'; b(2).EdgeColor = C.teal; b(2).LineWidth = 1.3;
    legend(b,{'Stereo measurement','Kalman filter'},'Location','best');
else
    text(ax,0.5,0.5,'Fewer than 2 paired interval samples','Units','normalized', ...
        'HorizontalAlignment','center','FontSize',8);
end
set(ax,'XTick',1:3,'XTickLabel',axisNames); xlim([0.5 3.5]);
ylabel('Position SD [cm]');
if stationaryIntervalConfirmed, intervalName = 'Stationary'; else, intervalName = 'Selected'; end
title(sprintf('(c) %s interval: %.1f-%.1f s',intervalName,precisionInterval_s));
formatAxes(ax,C);

ax = nexttile(L3); hold(ax,'on');
% A sample raster avoids implying precise dropout boundaries between samples.
plot(ax,t(valid),ones(nnz(valid),1),'.','Color',C.blue,'MarkerSize',5);
plot(ax,t(prediction),2*ones(nnz(prediction),1),'x','Color',C.orange,'MarkerSize',3);
plot(ax,t(~initialized),zeros(nnz(~initialized),1),'.','Color',C.grey,'MarkerSize',5);
set(ax,'YTick',[0 1 2],'YTickLabel',{'No KF','Stereo valid','Prediction only'});
ylim([-0.5 2.5]); xlim([t(1) t(end)]); xlabel('Time [s]');
title('(d) Measurement availability'); formatAxes(ax,C);

end % Optional additional figures

%% EXPORT: PNG ONLY (NO PDF)
if exportImages
    if ~isfolder(outputFolder), mkdir(outputFolder); end
    figures = positionFigures;
    names = positionNames;
    if createAdditionalFigures
        figures = [figures,{f2,f3}];
        names = [names,{'KF_02_velocity_and_speed.png','KF_03_filter_diagnostics.png'}];
    end
    for k = 1:numel(figures)
        exportgraphics(figures{k},fullfile(outputFolder,names{k}), ...
            'Resolution',resolutionDPI,'BackgroundColor','white');
    end
    writetable(dropoutSummary,fullfile(outputFolder,'dropout_summary.csv'));
    writetable(positionStats,fullfile(outputFolder,'position_difference_cm.csv'));
    writetable(velocityStats,fullfile(outputFolder,'velocity_difference_mps.csv'));
    writetable(precisionStats,fullfile(outputFolder,'selected_interval_SD_cm.csv'));
    fprintf('\nSaved %d PNG figures and statistics to:\n%s\n',numel(figures),outputFolder);
end

%% LOCAL FUNCTIONS
function f = reportFigure(name,width,height)
f = figure('Name',name,'Color','w','Units','centimeters', ...
    'Position',[2 2 width height],'NumberTitle','off');
end

function formatAxes(ax,C)
set(ax,'FontName','Arial','FontSize',9,'LineWidth',0.7,'TickDir','in', ...
    'Box','on','XColor',C.dark,'YColor',C.dark,'ZColor',C.dark, ...
    'GridLineStyle',':','GridAlpha',0.18,'MinorGridAlpha',0.07, ...
    'XMinorTick','on','YMinorTick','on','Layer','top');
grid(ax,'on'); ax.Title.FontWeight = 'normal'; ax.Title.FontSize = 10;
end

function [a,b,p] = comparisonPlot(ax,t,raw,kf,prediction,C,gap,target)
hold(ax,'on');
a = drawSeries(ax,t,raw,C.blue,'-','o',gap,target,0.55);
b = drawSeries(ax,t,kf,C.teal,'--','.',gap,target,1.15);
p = gobjects(0);
if any(prediction & isfinite(kf))
    ii = prediction & isfinite(kf);
    p = plot(ax,t(ii),kf(ii),'x','Color',C.orange,'MarkerSize',3.5,'LineWidth',0.8);
end
end

function comparisonLegend(ax,a,b,p,rawName,predictionName)
if isempty(p)
    legend(ax,[a b],{rawName,'Kalman filter'},'Location','best','FontSize',7);
else
    legend(ax,[a b p],{rawName,'Kalman filter',predictionName},'Location','best','FontSize',7);
end
end

function h = drawSeries(ax,t,y,colour,lineStyle,marker,gap,target,width)
[tt,yy] = insertGaps(t,y,gap);
finiteIndices = find(isfinite(yy));
if isempty(finiteIndices)
    markerIndices = [];
else
    step = max(1,ceil(numel(finiteIndices)/target));
    markerIndices = finiteIndices(1:step:end);
end
h = plot(ax,tt,yy,'Color',colour,'LineStyle',lineStyle,'LineWidth',width, ...
    'Marker',marker,'MarkerSize',3,'MarkerFaceColor','none', ...
    'MarkerIndices',markerIndices);
end

function [tt,yy] = insertGaps(t,y,gap)
% Insert NaN rows before long logging gaps without discarding endpoints.
cut = find(diff(t)>gap)+1;
n = numel(t); extra = zeros(n,1); extra(cut)=1;
destination = (1:n)'+cumsum(extra);
tt = nan(n+numel(cut),1); yy = nan(n+numel(cut),size(y,2));
tt(destination)=t; yy(destination,:)=y;
end

function S = differenceStats(d,axisNames)
n = zeros(3,1); bias = nan(3,1); spread = nan(3,1);
rmsd = nan(3,1); mad = nan(3,1); peak = nan(3,1);
for j = 1:3
    v = d(isfinite(d(:,j)),j); n(j)=numel(v);
    if n(j)>0
        bias(j)=mean(v); rmsd(j)=sqrt(mean(v.^2));
        mad(j)=mean(abs(v)); peak(j)=max(abs(v));
    end
    if n(j)>1, spread(j)=std(v,0); end
end
S = table(string(axisNames(:)),n,bias,spread,rmsd,mad,peak, ...
    'VariableNames',{'Axis','N','MeanDifference','SDDifference', ...
    'RMSDifference','MeanAbsDifference','MaxAbsDifference'});
end

function [S,bands,recoveryIndices] = dropoutEvents(t,mask,paired,raw,kf,gap)
% Split at non-dropout samples OR long timestamp gaps. Censored boundaries
% mean that a transition was not observed in the selected logged interval.
idx = find(mask);
starts = zeros(0,1); ends = zeros(0,1);
if ~isempty(idx)
    cut = [1;find(diff(idx)>1 | diff(t(idx))>gap)+1;numel(idx)+1];
    starts = idx(cut(1:end-1)); ends = idx(cut(2:end)-1);
end
n = numel(starts); bands = nan(n,2); recoveryIndices = zeros(0,1);
first = nan(n,1); last = first; span = first; duration = first;
count = zeros(n,1); leftCensored = true(n,1); rightCensored = true(n,1);
lastValid = nan(n,1); recoveryTime = lastValid; measurementGap = lastValid;
returnDifference = nan(n,3); returnNorm = nan(n,1);
for e = 1:n
    a=starts(e); b=ends(e);
    first(e)=t(a); last(e)=t(b); span(e)=t(b)-t(a); count(e)=b-a+1;
    bands(e,:)=[t(a),t(b)];
    leftObserved = a>1 && paired(a-1) && t(a)-t(a-1)<=gap;
    rightObserved = b<numel(t) && paired(b+1) && t(b+1)-t(b)<=gap;
    leftCensored(e)=~leftObserved; rightCensored(e)=~rightObserved;
    if leftObserved
        lastValid(e)=t(a-1);
        bands(e,1)=(t(a-1)+t(a))/2;
    end
    if rightObserved
        r=b+1; recoveryIndices(end+1,1)=r; %#ok<AGROW>
        recoveryTime(e)=t(r); bands(e,2)=(t(b)+t(r))/2;
        returnDifference(e,:)=raw(r,:)-kf(r,:);
        returnNorm(e)=norm(returnDifference(e,:));
    end
    if leftObserved && rightObserved
        duration(e)=bands(e,2)-bands(e,1);
        measurementGap(e)=recoveryTime(e)-lastValid(e);
    end
end
S = table((1:n)',first,last,count,span,duration,leftCensored,rightCensored, ...
    lastValid,recoveryTime,measurementGap,returnDifference(:,1), ...
    returnDifference(:,2),returnDifference(:,3),returnNorm, ...
    'VariableNames',{'DropoutID','FirstMissingLog_s','LastMissingLog_s', ...
    'MissingLogCount','MissingLogSpan_s','EstimatedDuration_s', ...
    'StartCensored','EndCensored','LastValidLog_s','FirstReturnLog_s', ...
    'ValidLogSeparation_s','ReturnPostUpdateDifferenceX_cm', ...
    'ReturnPostUpdateDifferenceY_cm','ReturnPostUpdateDifferenceZ_cm', ...
    'ReturnPostUpdateDifference3D_cm'});
end

function markDropouts(ax,bands,C,showLabels)
% Keep y limits fixed while adding full-height translucent backgrounds.
% Zero-span censored single samples are shown by a vertical dotted line.
hold(ax,'on'); limits=ylim(ax);
for e=1:size(bands,1)
    a=bands(e,1); b=bands(e,2);
    if b>a
        h=patch(ax,[a b b a],limits([1 1 2 2]),C.orange, ...
            'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
        uistack(h,'bottom');
    else
        xline(ax,a,':','Color',C.orange,'HandleVisibility','off');
    end
    if showLabels
        text(ax,(a+b)/2,limits(2)-0.03*diff(limits),sprintf('D%d',e), ...
            'HorizontalAlignment','center','VerticalAlignment','top', ...
            'Color',C.orange,'FontSize',7,'Clipping','on');
    end
end
ylim(ax,limits);
end
