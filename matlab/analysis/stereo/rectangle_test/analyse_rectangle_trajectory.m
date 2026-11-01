%% analyse_rectangle_trajectory.m
% Analysis of a known rectangular drone trajectory recorded by the stereo
% LED tracking system.
%
% Rectangle dimensions:
%   Long side  = 24.0 cm
%   Short side = 12.6 cm
%
% The script:
%   1. Filters invalid triangle solutions.
%   2. Selects a defined analysis time interval.
%   3. Reports tracking/detection quality.
%   4. Plots the 3-D trajectory.
%   5. Plots X, Y and Z positions over time.
%   6. Plots the 3-D trajectory with time represented by colour.
%
% No Statistics and Machine Learning Toolbox is required.

clear;
close all;
clc;


%% ------------------------- USER SETTINGS -------------------------------

scriptFolder = fileparts(mfilename('fullpath'));

csvFile = fullfile(scriptFolder, ...
    'triangle_tracking_tof_20260803_115949.csv');

rectangleLong_cm  = 24.0;
rectangleShort_cm = 12.6;


% ---------------------- ANALYSIS TIME INTERVAL --------------------------
% Only data between these two times is used for analysis and plotting.

startTime_s = 10;
endTime_s   = 130;


% Position used as the drone reference point.

xColumn = 'rear_center_X_m';
yColumn = 'rear_center_Y_m';
zColumn = 'rear_center_Z_m';


% A new trajectory segment is started after this tracking gap.
% This avoids drawing misleading lines between positions separated by
% invalid data.

maximumConnectionGap_s = 0.50;


% Expected LED triangle dimensions from the physical drone setup [cm].
% Set any value to NaN if it should not be compared.

expectedRearBase_cm   = 12.0;
expectedFrontLeft_cm  = 8.3;
expectedFrontRight_cm = 7.3;


% Plausible ToF range.
% 32766 mm is treated as an invalid/sentinel reading.

maximumPlausibleToF_mm = 4000;


outputFolder = fullfile( ...
    scriptFolder, ...
    'rectangle_analysis_figures');


%% ---------------------- COLOUR STANDARD --------------------------------
% Colours taken from Farben.m.
%
% According to the convention:
%
% rr1 = measured output 1
% rr2 = measured output 2
% rr3 = measured output 3
%
% Therefore:
%
% X position -> rr1
% Y position -> rr2
% Z position -> rr3

rr1 = [255   0   0] / 255;     % red
rr2 = [128   0   0] / 255;     % dark red
rr3 = [240 128 128] / 255;     % light red

% University dark grey, useful for auxiliary trajectory lines.
hg1 = [110 111 110] / 255;


% Continuous time colour map based only on the measured-output red family:
%
% beginning -> light red
% middle    -> red
% end       -> dark red

colourMapTime = interp1( ...
    [0; 0.5; 1], ...
    [rr3; rr1; rr2], ...
    linspace(0, 1, 256)', ...
    'linear');


%% -------------------------- LOAD DATA ----------------------------------

if ~isfile(csvFile)

    error([ ...
        'CSV file not found: ', csvFile, newline, ...
        'Place the MATLAB script in the same folder as the CSV file, ', ...
        'or provide the full path in csvFile.' ...
        ]);

end


T = readtable( ...
    csvFile, ...
    'VariableNamingRule', 'preserve');


requiredColumns = { ...
    'time_s', ...
    'triangle_valid', ...
    'triangle_detected', ...
    'reject_reason', ...
    xColumn, ...
    yColumn, ...
    zColumn};


missingColumns = requiredColumns( ...
    ~ismember( ...
        requiredColumns, ...
        T.Properties.VariableNames));


if ~isempty(missingColumns)

    error( ...
        'Missing required columns: %s', ...
        strjoin(missingColumns, ', '));

end


if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


%% -------------------- EXTRACT RAW DATA ---------------------------------

timeAll_s = T.time_s;

positionAll_m = [ ...
    T.(xColumn), ...
    T.(yColumn), ...
    T.(zColumn)];


%% ---------------------- TIME INTERVAL ----------------------------------
% Logical mask:
%
% true  -> measurement lies inside 40...85 s
% false -> measurement lies outside this interval

insideTimeInterval = ...
    timeAll_s >= startTime_s & ...
    timeAll_s <= endTime_s;


%% ---------------------- VALID POSITION DATA ----------------------------
% Invalid rows in this recording often contain exactly -1 m.
%
% A sample is accepted only when:
%
%   1. It lies inside the selected time interval.
%   2. triangle_valid == 1.
%   3. XYZ values are finite.
%   4. XYZ values are not the invalid -1 m sentinel.

validPosition = ...
    insideTimeInterval & ...
    T.triangle_valid == 1 & ...
    all(isfinite(positionAll_m), 2) & ...
    all(positionAll_m > -0.90, 2);


%% ---------------------- SELECT DATA ------------------------------------

time_s = timeAll_s(validPosition);

position_cm = ...
    100 .* positionAll_m(validPosition, :);


if size(position_cm, 1) < 3

    error([ ...
        'Fewer than three valid position samples are available ', ...
        'between %.1f s and %.1f s.' ...
        ], ...
        startTime_s, ...
        endTime_s);

end


X_cm = position_cm(:, 1);
Y_cm = position_cm(:, 2);
Z_cm = position_cm(:, 3);


%% -------------------- TRACKING STATISTICS ------------------------------

% Number of recorded rows inside the selected interval.
numberOfRows = nnz(insideTimeInterval);

% Number of valid position measurements inside the selected interval.
numberValid = nnz(validPosition);

validPercentage = ...
    100 * numberValid / numberOfRows;


fprintf('\n');
fprintf('------------------------------------------\n');
fprintf('Analysis interval\n');
fprintf('------------------------------------------\n');

fprintf( ...
    'Start time       : %.2f s\n', ...
    startTime_s);

fprintf( ...
    'End time         : %.2f s\n', ...
    endTime_s);

fprintf( ...
    'Samples in range : %d\n', ...
    numberOfRows);

fprintf( ...
    'Valid samples    : %d\n', ...
    numberValid);

fprintf( ...
    'Valid percentage : %.2f %%\n', ...
    validPercentage);

fprintf('------------------------------------------\n\n');


intervalText = sprintf( ...
    '%.0f-%.0f s', ...
    startTime_s, ...
    endTime_s);
disp('mean=')
mean(Z_cm)

%% ======================================================================
% PLOT 1
% 3-D TRAJECTORY
% =======================================================================

figure1 = figure( ...
    'Color', 'w', ...
    'Position', [100, 100, 900, 700]);


plot3( ...
    X_cm, ...
    Y_cm, ...
    Z_cm, ...
    '-', ...
    'Color', rr1, ...
    'LineWidth', 1.5);


grid on;
axis equal;


xlabel('X [cm]');
ylabel('Y [cm]');
zlabel('Z [cm]');


title( ...
    '3-D Drone Manual Trajectory');


view(35, 25);


%% ======================================================================
% PLOT 2
% X, Y AND Z AS SEPARATE SUBPLOTS
% =======================================================================

figure2 = figure( ...
    'Color', 'w', ...
    'Position', [120, 80, 1000, 800]);


% ----------------------------- X ----------------------------------------

subplot(3, 1, 1);

plot( ...
    time_s, ...
    X_cm, ...
    '-', ...
    'Color', rr1, ...
    'LineWidth', 1.4);

grid on;

xlim([startTime_s, endTime_s]);

xlabel('Time [s]');
ylabel('X [cm]');

title('X Position over Time');


% ----------------------------- Y ----------------------------------------

subplot(3, 1, 2);

plot( ...
    time_s, ...
    Y_cm, ...
    '-', ...
    'Color', rr2, ...
    'LineWidth', 1.4);

grid on;

xlim([startTime_s, endTime_s]);

xlabel('Time [s]');
ylabel('Y [cm]');

title('Y Position over Time');


% ----------------------------- Z ----------------------------------------

subplot(3, 1, 3);

plot( ...
    time_s, ...
    Z_cm, ...
    '-', ...
    'Color', rr3, ...
    'LineWidth', 1.4);

grid on;

xlim([startTime_s, endTime_s]);

xlabel('Time [s]');
ylabel('Z [cm]');

title('Z Position over Time');


%% ======================================================================
% PLOT 3
% X, Y AND Z IN ONE FIGURE
% =======================================================================

figure3 = figure( ...
    'Color', 'w', ...
    'Position', [140, 100, 1000, 650]);


plot( ...
    time_s, ...
    X_cm, ...
    '-', ...
    'Color', rr1, ...
    'LineWidth', 1.4);

hold on;


plot( ...
    time_s, ...
    Y_cm, ...
    '-', ...
    'Color', rr2, ...
    'LineWidth', 1.4);


plot( ...
    time_s, ...
    Z_cm, ...
    '-', ...
    'Color', rr3, ...
    'LineWidth', 1.4);


grid on;

xlim([startTime_s, endTime_s]);


xlabel('Time [s]');
ylabel('Position [cm]');


legend( ...
    'X', ...
    'Y', ...
    'Z', ...
    'Location', 'best');


title( ...
    ['X, Y and Z Position, ', intervalText]);


%% ======================================================================
% PLOT 4
% 3-D TRAJECTORY COLOURED BY TIME
% =======================================================================

figure4 = figure( ...
    'Color', 'w', ...
    'Position', [160, 100, 900, 700]);


scatter3( ...
    X_cm, ...
    Y_cm, ...
    Z_cm, ...
    25, ...
    time_s, ...
    'filled');


grid on;
axis equal;


xlabel('X [cm]');
ylabel('Y [cm]');
zlabel('Z [cm]');


title( ...
    ['3-D Drone Trajectory Coloured by Time, ', intervalText]);


% Apply the red-family colour map defined above.
colormap(gca, colourMapTime);


% Explicitly use the chosen time interval for the colour scale.
clim([startTime_s, endTime_s]);


colourBar = colorbar;

colourBar.Label.String = ...
    'Time [s]';


view(35, 25);


%% ======================================================================
% PLOT 5
% 3-D TIME COLOUR + CONNECTING TRAJECTORY
% =======================================================================

figure5 = figure( ...
    'Color', 'w', ...
    'Position', [180, 100, 900, 700]);


% Coloured measurement points.
scatter3( ...
    X_cm, ...
    Y_cm, ...
    Z_cm, ...
    25, ...
    time_s, ...
    'filled');


hold on;


% Auxiliary connecting line.
% Dark grey is used so that it does not compete with the measured-output
% colour representation.

plot3( ...
    X_cm, ...
    Y_cm, ...
    Z_cm, ...
    '-', ...
    'Color', hg1, ...
    'LineWidth', 0.8);


grid on;
axis equal;


xlabel('X [cm]');
ylabel('Y [cm]');
zlabel('Z [cm]');


title( ...
    '3-D Rectangular Drone Trajectory (Manual)');


colormap(gca, colourMapTime);

clim([startTime_s, endTime_s]);


colourBar = colorbar;

colourBar.Label.String = ...
    'Time [s]';


view(35, 25);