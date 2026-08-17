function Autonomous_Microgrid_Platform()
% =========================================================================
% DETAILED AUTONOMOUS MICROGRID EMS
% 1. Pre-Flight: base Simulink only
% 2. Fetches NASA API Data & Preprocesses Inputs
% 3. Trains 7 AI Models & Evaluates Trade-offs (normalized scoring,
%    including per-model Accuracy % and R^2)
% 4. Auto-Selects Best Model for AI-Scheduling Control
% 5. Programmatically Builds a FULLY WIRED plain-Simulink Power-Flow
%    Topology
% 6. Launches Multi-Tab Web UI with Real-Time Component States AND a
%    dedicated live telemetry tab that charts every single signal that
%    feeds the Simulink model (all 11 input channels), not just the
%    dispatch/voltage summary charts.
% =========================================================================
    clc; clear; close all;
    
    % Generate a single timestamp for all saved files and models in this run
    runTimestamp = datestr(now, 'yyyymmdd_HHMMSS');
    
    disp('Initializing Detailed Autonomous Microgrid Platform...');
    
    %% PRE-FLIGHT CHECK: BASE SIMULINK ONLY
    disp('--- Pre-Flight: Loading Simulink ---');
    try
        load_system('simulink');
        disp('SUCCESS: Simulink initialized.');
    catch ME
        warning('Failed to open Simulink. The script cannot proceed without it.');
        disp(ME.message);
        return;
    end
    
    %% PHASE 1: DATA ACQUISITION & PREPROCESSING
    disp('--- Phase 1: Fetching & Preprocessing Online Datasets ---');
    % Tongi Industrial Zone Coordinates
    lat = 23.9167; lon = 90.3667;
    startDate = '20210101'; endDate = '20251231';
    apiUrl = sprintf('https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,WS10M,T2M,RH2M,PS&community=RE&longitude=%.4f&latitude=%.4f&start=%s&end=%s&format=JSON', lon, lat, startDate, endDate);
    
    try
        disp('Querying NASA POWER API...');
        weatherData = webread(apiUrl);
        disp('Data retrieved successfully.');
    catch
        error('API Connection failed. Ensure you have an active internet connection.');
    end
    
    % Extract Features and force them into Column Vectors
    raw_G  = cell2mat(struct2cell(weatherData.properties.parameter.ALLSKY_SFC_SW_DWN)); raw_G = raw_G(:);
    raw_W  = cell2mat(struct2cell(weatherData.properties.parameter.WS10M)); raw_W = raw_W(:);
    raw_T  = cell2mat(struct2cell(weatherData.properties.parameter.T2M)); raw_T = raw_T(:);
    raw_RH = cell2mat(struct2cell(weatherData.properties.parameter.RH2M)); raw_RH = raw_RH(:);
    raw_PS = cell2mat(struct2cell(weatherData.properties.parameter.PS)); raw_PS = raw_PS(:);
    
    % Data cleaning: anomalous or missing sensor data
    raw_G(raw_G < 0) = 0;
    raw_W(raw_W < 0) = mean(raw_W(raw_W >= 0));
    raw_T(raw_T < -50 | raw_T > 60) = mean(raw_T(raw_T >= -50 & raw_T <= 60));
    raw_RH(raw_RH < 0 | raw_RH > 100) = mean(raw_RH(raw_RH >= 0 & raw_RH <= 100));
    raw_PS(raw_PS < 50 | raw_PS > 110) = mean(raw_PS(raw_PS >= 50 & raw_PS <= 110));
    
    % Use the FULL dataset (all downloaded timesteps)
    N = length(raw_G);
    
    % Normalized Load Demand (50 kW to 65 kW)
    raw_Demand = 45e3 + (20e3 * (0.5 + 0.4*sin(2*pi*(1:N)'/24) + 0.1*randn(N,1)));
    
    % Normalize [0,1] function
    minMax = @(x) (x - min(x)) ./ (max(x) - min(x) + eps);
    
    % Assemble X_matrix skeleton (11 x N)
    X_matrix = [minMax(raw_G(1:N)), minMax(raw_W(1:N)), minMax(raw_Demand), minMax(raw_T(1:N)), ...
                minMax(raw_RH(1:N)), minMax(raw_PS(1:N)), zeros(N,1), minMax(mod((1:N)', 24)), ...
                minMax(sin(2*pi*(1:N)'/8760)), zeros(N,1), zeros(N,1)]';
            
    % Rule-Based Heuristic Generation Variables
    Y_matrix = zeros(3, N);
    battery_SoC_array = zeros(1, N);
    raw_imbalance = zeros(1, N);
    prev_action = zeros(1, N);
    battery_SoC = 0.5;
    
    for t = 1:N
        P_req = raw_Demand(t);
        P_pv = min(P_req, (raw_G(t)/1000) * 80e3);
        P_wt = min(P_req - P_pv, (raw_W(t)/12)^3 * 30e3);
        P_rem = P_req - (P_pv + P_wt);
        
        % Track the ACTUAL battery discharge amount explicitly
        P_bat_disp = 0;
        if P_rem > 0
            if battery_SoC > 0.3
                P_bat_disp = min(P_rem, 40e3);
                P_rem = P_rem - P_bat_disp;
            end
            P_fc = min(50e3, max(0, P_rem));
            if t < N && P_bat_disp > 0
                battery_SoC = max(0.5, battery_SoC - (P_bat_disp * 1e-6));
            end
        else
            P_fc = 0;
            P_surplus = abs(P_rem);
            if t < N
                battery_SoC = min(0.9, battery_SoC + (P_surplus * 1e-6));
            end
        end
        
        % Store values for normalized injection
        battery_SoC_array(t) = battery_SoC;
        raw_imbalance(t) = (P_pv + P_wt + P_fc) - P_req;
        if t > 1
            prev_action(t) = mean(Y_matrix(:, t-1));
        end
        Y_matrix(:, t) = [P_pv/80e3; P_fc/50e3; P_wt/30e3];
    end
    
    % Inject arrays into X_matrix
    X_matrix(7, :) = battery_SoC_array;
    X_matrix(10, :) = minMax(raw_imbalance');
    X_matrix(11, :) = prev_action;
    
    % Overlapping windows for proper mini-batch training
    seqLen = 50;      % length of each training window (timesteps)
    seqStride = 5;    % hop between window start points (overlap = seqLen-seqStride)
    numSeq = max(1, floor((N - seqLen) / seqStride) + 1);
    XSeq = cell(numSeq, 1); YSeq = cell(numSeq, 1);
    
    for s = 1:numSeq
        idxStart = (s - 1) * seqStride + 1;
        idxEnd = idxStart + seqLen - 1;
        XSeq{s} = X_matrix(:, idxStart:idxEnd);
        YSeq{s} = Y_matrix(:, idxStart:idxEnd);
    end
    
    rng(42); % reproducible train/validation split
    valFraction = 0.15;
    numVal = max(1, round(valFraction * numSeq));
    shuffledIdx = randperm(numSeq);
    valIdx = shuffledIdx(1:numVal);
    trainIdx = shuffledIdx(numVal+1:end);
    
    XTrain = XSeq(trainIdx); YTrain = YSeq(trainIdx);
    XVal = XSeq(valIdx);     YVal = YSeq(valIdx);
    
    fprintf('Training set: %d windows (%d timesteps each) | Validation set: %d windows\n', ...
        numel(XTrain), seqLen, numel(XVal));
    disp('Dataset preprocessing and labeling complete.');
    
    % =========================================================================
    % CSV EXPORT: Complete AI Model Input Data & Targets
    % =========================================================================
    inputColNames_X = {'Norm_Solar_Irradiance', 'Norm_Wind_Speed', 'Norm_Load_Demand', ...
                       'Norm_Temperature', 'Norm_Relative_Humidity', 'Norm_Surface_Pressure', ...
                       'Battery_SOC_Input', 'Norm_Time_Index', 'Norm_Season_Index', ...
                       'Norm_Power_Imbalance', 'Previous_Action', ...
                       'Target_PV_Fraction', 'Target_FC_Fraction', 'Target_WT_Fraction'};
                       
    % Transpose matrices to get Nx11 inputs and Nx3 targets, combine them into an Nx14 array
    aiDatasetTable = array2table([X_matrix', Y_matrix'], 'VariableNames', inputColNames_X);
    aiDatasetCsvName = sprintf('AI_Model_Training_Dataset_%s.csv', runTimestamp);
    
    try
        writetable(aiDatasetTable, aiDatasetCsvName);
        fprintf('Complete AI model training dataset (Inputs + Targets) exported to: %s\n', aiDatasetCsvName);
    catch ME
        warning('Could not write AI training dataset CSV (%s).', ME.message);
    end
    % =========================================================================

    %% PHASE 2: MULTI-MODEL AI TRAINING & BENCHMARKING
    disp('--- Phase 2: Training & Benchmarking AI Architectures ---');
    % Hardware Detection
    execEnv = 'auto';
    try
        if exist('gcp', 'file') == 2
            poolObj = gcp('nocreate');
            if isempty(poolObj), parpool; end
            numGPUs = gpuDeviceCount("available");
            if numGPUs > 1
                execEnv = 'multi-gpu';
            elseif numGPUs == 1
                execEnv = 'gpu';
            else
                execEnv = 'parallel';
            end
        end
    catch
        disp('Parallel functionality restricted; defaulting to standard execution.');
    end
    
    numFeatures = 11; numResponses = 3; numHiddenUnits = 96; lagWindowSize = 5;
    
    % Define Architectures
    archs.GRU = [sequenceInputLayer(numFeatures), gruLayer(numHiddenUnits, 'OutputMode', 'sequence'), fullyConnectedLayer(numResponses), sigmoidLayer(), regressionLayer()];
    archs.LSTM = [sequenceInputLayer(numFeatures), lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'), fullyConnectedLayer(numResponses), sigmoidLayer(), regressionLayer()];
    archs.StackedLSTM = [sequenceInputLayer(numFeatures), lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'), lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'), fullyConnectedLayer(numResponses), sigmoidLayer(), regressionLayer()];
    archs.FNN = [sequenceInputLayer(numFeatures), fullyConnectedLayer(numHiddenUnits), reluLayer(), fullyConnectedLayer(numResponses), sigmoidLayer(), regressionLayer()];
    archs.NARX_FNN = [sequenceInputLayer(numFeatures), convolution1dLayer(lagWindowSize, numHiddenUnits, 'Padding', 'causal'), reluLayer(), fullyConnectedLayer(numResponses), sigmoidLayer(), regressionLayer()];
    archs.CNNLSTM = [sequenceInputLayer(numFeatures), convolution1dLayer(3, 32, 'Padding', 'causal'), reluLayer(), lstmLayer(numHiddenUnits, 'OutputMode', 'sequence'), fullyConnectedLayer(numResponses), sigmoidLayer(), regressionLayer()];
    archs.BiLSTM = [sequenceInputLayer(numFeatures), bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence'), fullyConnectedLayer(numResponses), sigmoidLayer(), regressionLayer()];
    modelNames = fieldnames(archs);
    
    opts = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 16, ...
        'InitialLearnRate', 0.005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.5, ...
        'L2Regularization', 1e-4, ...
        'GradientThreshold', 1, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XVal, YVal}, ...
        'ValidationFrequency', max(1, floor(numel(XTrain) / 16)), ...
        'ValidationPatience', 15, ...
        'OutputNetwork', 'best-validation-loss', ...
        'Verbose', 0, 'Plots', 'none', 'ExecutionEnvironment', execEnv);
        
    netList = cell(1, length(modelNames));
    rmseList = zeros(1, length(modelNames));
    latencyList = zeros(1, length(modelNames));
    r2List = zeros(1, length(modelNames));
    accuracyList = zeros(1, length(modelNames));
    
    fprintf('\n%-15s | %-10s | %-12s | %-10s | %-18s\n', 'Model', 'RMSE', 'Accuracy %', 'R^2', 'Latency (ms/step)');
    fprintf('----------------------------------------------------------------------------\n');
    
    for i = 1:length(modelNames)
        mName = modelNames{i};
        net = trainNetwork(XTrain, YTrain, archs.(mName), opts);
        
        % Measure Latency
        testInput = rand(numFeatures, 1);
        tic; for k=1:500, predict(net, testInput); end
        latency = (toc / 500) * 1000;
        
        % Measure RMSE/Accuracy/R^2 on the FULL original timeline
        pred = predict(net, X_matrix);
        rmse = sqrt(mean((Y_matrix - pred).^2, 'all'));
        accuracy_pct = max(0, min(100, (1 - rmse) * 100));
        ss_res = sum((Y_matrix - pred).^2, 'all');
        ss_tot = sum((Y_matrix - mean(Y_matrix, 'all')).^2, 'all');
        r2 = 1 - (ss_res / (ss_tot + eps));
        
        netList{i} = net;
        rmseList(i) = rmse;
        latencyList(i) = latency;
        accuracyList(i) = accuracy_pct;
        r2List(i) = r2;
        fprintf('%-15s | %-10.5f | %-12.2f | %-10.4f | %-18.4f\n', mName, rmse, accuracy_pct, r2, latency);
    end
    fprintf('----------------------------------------------------------------------------\n');
    
    normRMSE = (rmseList - min(rmseList)) ./ (max(rmseList) - min(rmseList) + eps);
    normLatency = (latencyList - min(latencyList)) ./ (max(latencyList) - min(latencyList) + eps);
    w_rmse = 0.4; w_latency = 0.6; 
    scoreList = (w_rmse * normRMSE) + (w_latency * normLatency);
    
    [bestScore, bestIdx] = min(scoreList); %#ok<ASGLU>
    bestNet = netList{bestIdx};
    bestModelName = modelNames{bestIdx};
    bestLatency = latencyList(bestIdx);
    bestAccuracy = accuracyList(bestIdx);
    bestR2 = r2List(bestIdx);
    
    fprintf('>>> BEST MODEL SELECTED: %s (normalized score = %.4f, Accuracy = %.2f%%, R^2 = %.4f) <<<\n', ...
        bestModelName, bestScore, bestAccuracy, bestR2);
    save('Best_Microgrid_Scheduler.mat', 'bestNet');
    
    benchmarkCsvName = sprintf('AI_Model_Benchmark_Results_%s.csv', runTimestamp);
    isSelected = false(length(modelNames), 1);
    isSelected(bestIdx) = true;
    benchmarkTable = table(modelNames, rmseList(:), accuracyList(:), r2List(:), latencyList(:), ...
        normRMSE(:), normLatency(:), scoreList(:), isSelected, ...
        'VariableNames', {'Model_Name', 'RMSE', 'Accuracy_Pct', 'R_Squared', 'Latency_ms_per_step', ...
                           'Normalized_RMSE', 'Normalized_Latency', 'Combined_Score', 'Selected_As_Best'});
    try
        writetable(benchmarkTable, benchmarkCsvName);
        fprintf('AI model benchmarking results exported to: %s\n', benchmarkCsvName);
    catch ME
        warning('Could not write benchmark CSV (%s).', ME.message);
    end
    
    %% PHASE 3: FULLY WIRED PLAIN-SIMULINK POWER-FLOW MODEL
    disp('--- Phase 3: Building Fully Wired Power-Flow Topology (Simulink Model) ---');
    mdl = sprintf('Hybrid_Microgrid_EMS');
    new_system(mdl); open_system(mdl);
    inputs = {'Solar_Irradiance', 'Wind_Speed', 'Load_Demand', 'Temperature', ...
              'Relative_Humidity', 'Surface_Pressure', 'Battery_SOC', ...
              'Time_Index', 'Season_Index', 'Power_Imbalance', 'Previous_Action'};
              
    % --- 1. AI Supervisory Layer input mux ---
    add_block('simulink/Signal Routing/Mux', [mdl '/Features_Mux'], 'Inputs', '11', 'Position', [200, 50, 210, 380]);
    for k = 1:length(inputs)
        y_pos = 20 + (30 * k);
        add_block('simulink/Sources/In1', [mdl '/' inputs{k}], 'Position', [50, y_pos, 80, y_pos+15]);
        add_line(mdl, [inputs{k} '/1'], ['Features_Mux/' num2str(k)], 'autorouting', 'on');
    end
    
    aiBlockName = ['AI_Scheduler_' bestModelName];
    aiBlockPath = [mdl '/' aiBlockName];
    add_block('simulink/User-Defined Functions/MATLAB Function', aiBlockPath, 'Position', [280, 160, 430, 270]);
    add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '3', 'Position', [480, 150, 485, 280]);
    
    matlabFnCode = sprintf([ ...
        'function y = predictAI(u)\n' ...
        '%%#codegen\n' ...
        'coder.extrinsic(''predict'',''load'');\n' ...
        'persistent net\n' ...
        'if isempty(net)\n' ...
        '    s = load(''Best_Microgrid_Scheduler.mat'', ''bestNet'');\n' ...
        '    net = s.bestNet;\n' ...
        'end\n' ...
        'y = zeros(3,1);\n' ...
        'y = predict(net, u);\n' ...
        'end\n' ...
    ]);
    try
        rt = sfroot;
        chartObj = rt.find('-isa', 'Stateflow.EMChart', 'Path', aiBlockPath);
        if isempty(chartObj)
            machine = sfroot.find('-isa', 'Stateflow.Machine', 'Name', mdl);
            charts = machine.find('-isa', 'Stateflow.EMChart');
            for c = 1:numel(charts)
                if strcmp(charts(c).Path, aiBlockPath)
                    chartObj = charts(c);
                    break;
                end
            end
        end
        chartObj.Script = matlabFnCode;
        disp(['AI Scheduler block wired to live network: ' bestModelName]);
    catch ME
        warning('Could not auto-populate the MATLAB Function block script (%s). Open "%s" in the model and paste the predictor code below manually:', ME.message, aiBlockPath);
        disp(matlabFnCode);
    end
    
    add_line(mdl, 'Features_Mux/1', [aiBlockName '/1'], 'autorouting', 'on');
    add_line(mdl, [aiBlockName '/1'], 'Demux/1', 'autorouting', 'on');
    
    add_block('simulink/Math Operations/Gain', [mdl '/PV_Capacity_80kW'], 'Gain', '80', 'Position', [550, 60, 610, 90]);
    add_block('simulink/Math Operations/Gain', [mdl '/FC_Capacity_50kW'], 'Gain', '50', 'Position', [550, 190, 610, 220]);
    add_block('simulink/Math Operations/Gain', [mdl '/WT_Capacity_30kW'], 'Gain', '30', 'Position', [550, 320, 610, 350]);
    add_block('simulink/Discontinuities/Saturation', [mdl '/PV_Sat'], 'LowerLimit', '0', 'UpperLimit', '80', 'Position', [650, 60, 700, 90]);
    add_block('simulink/Discontinuities/Saturation', [mdl '/FC_Sat'], 'LowerLimit', '0', 'UpperLimit', '50', 'Position', [650, 190, 700, 220]);
    add_block('simulink/Discontinuities/Saturation', [mdl '/WT_Sat'], 'LowerLimit', '0', 'UpperLimit', '30', 'Position', [650, 320, 700, 350]);
    
    add_line(mdl, 'Demux/1', 'PV_Capacity_80kW/1', 'autorouting', 'on');
    add_line(mdl, 'Demux/2', 'FC_Capacity_50kW/1', 'autorouting', 'on');
    add_line(mdl, 'Demux/3', 'WT_Capacity_30kW/1', 'autorouting', 'on');
    add_line(mdl, 'PV_Capacity_80kW/1', 'PV_Sat/1', 'autorouting', 'on');
    add_line(mdl, 'FC_Capacity_50kW/1', 'FC_Sat/1', 'autorouting', 'on');
    add_line(mdl, 'WT_Capacity_30kW/1', 'WT_Sat/1', 'autorouting', 'on');
    
    add_block('simulink/Math Operations/Sum', [mdl '/Generation_Sum'], 'Inputs', '+++', 'Position', [740, 150, 770, 260]);
    add_line(mdl, 'PV_Sat/1', 'Generation_Sum/1', 'autorouting', 'on');
    add_line(mdl, 'FC_Sat/1', 'Generation_Sum/2', 'autorouting', 'on');
    add_line(mdl, 'WT_Sat/1', 'Generation_Sum/3', 'autorouting', 'on');
    
    add_block('simulink/Sources/In1', [mdl '/Load_Demand_kW'], 'Position', [740, 400, 770, 420]);
    add_block('simulink/Math Operations/Sum', [mdl '/Power_Mismatch'], 'Inputs', '+-', 'Position', [820, 200, 850, 260]);
    add_line(mdl, 'Generation_Sum/1', 'Power_Mismatch/1', 'autorouting', 'on');
    add_line(mdl, 'Load_Demand_kW/1', 'Power_Mismatch/2', 'autorouting', 'on');
    
    add_block('simulink/Math Operations/Gain', [mdl '/SoC_Rate_Gain'], 'Gain', '1e-3', 'Position', [900, 350, 930, 380]);
    add_block('simulink/Continuous/Integrator', [mdl '/Battery_SoC_Integrator'], 'InitialCondition', '0.5', 'Position', [980, 350, 1010, 380]);
    add_block('simulink/Discontinuities/Saturation', [mdl '/SoC_Saturation'], 'LowerLimit', '0.2', 'UpperLimit', '0.9', 'Position', [1040, 350, 1090, 380]);
    
    add_line(mdl, 'Power_Mismatch/1', 'SoC_Rate_Gain/1', 'autorouting', 'on');
    add_line(mdl, 'SoC_Rate_Gain/1', 'Battery_SoC_Integrator/1', 'autorouting', 'on');
    add_line(mdl, 'Battery_SoC_Integrator/1', 'SoC_Saturation/1', 'autorouting', 'on');
    
    add_block('simulink/Sinks/Scope', [mdl '/Power_Dispatch_Scope'], 'Position', [1150, 150, 1190, 200]);
    add_block('simulink/Sinks/Scope', [mdl '/Battery_SoC_Scope'], 'Position', [1150, 350, 1190, 400]);
    add_line(mdl, 'Power_Mismatch/1', 'Power_Dispatch_Scope/1', 'autorouting', 'on');
    add_line(mdl, 'SoC_Saturation/1', 'Battery_SoC_Scope/1', 'autorouting', 'on');
    
    disp('Fully wired plain-Simulink power-flow topology generated (no Simscape required).');
    save_system(mdl);
    disp(['Simulink model generation complete: ' mdl '.slx']);
    
    %% PHASE 4: MICROGRID DASHBOARD (WITH GRID LOGIC)
    disp('--- Phase 4: Launching Real-Time Monitoring Dashboard ---');
    appFig = uifigure('Name', 'Autonomous Microgrid Dashboard', 'Position', [100 50 1150 750], 'Color', '#F0F2F5');
    uilabel(appFig, 'Text', 'Autonomous On-Grid Hybrid Microgrid EMS', ...
        'Position', [300 700 550 40], 'FontSize', 22, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    tg = uitabgroup(appFig, 'Position', [20 20 1110 670]);
    tab1 = uitab(tg, 'Title', 'Real-Time AI Driven Microgrid Dashboard');
    
    pnlStates = uipanel(tab1, 'Title', 'Real-Time System States', 'Position', [20 20 320 600], 'FontSize', 14, 'FontWeight', 'bold');
    lblGridState = uilabel(pnlStates, 'Text', 'GRID: INITIALIZING...', 'Position', [10 540 300 35], 'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', '#4DBEEE', 'FontColor', 'w');
    lblAI = uilabel(pnlStates, 'Text', sprintf('AI Scheduler: %s (%.2f ms)', bestModelName, bestLatency), 'Position', [15 495 290 30], 'FontSize', 13, 'FontWeight', 'bold');
    lblAcc = uilabel(pnlStates, 'Text', sprintf('Model Accuracy: %.2f%%  |  R^2: %.4f', bestAccuracy, bestR2), 'Position', [15 465 290 30], 'FontSize', 12, 'FontColor', '#0072BD');
    lblLoad = uilabel(pnlStates, 'Text', 'Tongi Area Load: 0.0 kW', 'Position', [15 425 290 30], 'FontSize', 13, 'FontWeight', 'bold');
    lblTotalGen = uilabel(pnlStates, 'Text', 'Total Generation: 0.0 kW', 'Position', [15 390 290 30], 'FontSize', 13, 'FontWeight', 'bold', 'FontColor', '#A2142F');
    lblPV = uilabel(pnlStates, 'Text', 'PV Energy: 0.0 kW (STANDBY)', 'Position', [15 350 290 30], 'FontSize', 13);
    lblWT = uilabel(pnlStates, 'Text', 'Wind Turbine: 0.0 kW (STANDBY)', 'Position', [15 310 290 30], 'FontSize', 13);
    lblFC = uilabel(pnlStates, 'Text', 'Fuel Energy: 0.0 kW (STANDBY)', 'Position', [15 270 290 30], 'FontSize', 13);
    lblBESS = uilabel(pnlStates, 'Text', 'Battery SOC: 50.0% (IDLE)', 'Position', [15 220 290 30], 'FontSize', 13);
    lblDCDC = uilabel(pnlStates, 'Text', 'DC-DC Strategy: IDLE', 'Position', [15 180 290 30], 'FontSize', 13);
    
    axPower = uiaxes(tab1, 'Position', [360 320 720 290]);
    title(axPower, 'Power Dispatch vs Industrial Demand');
    xlabel(axPower, 'Simulation Step'); ylabel(axPower, 'Power (kW)');
    grid(axPower, 'on'); hold(axPower, 'on');
    
    axVoltage = uiaxes(tab1, 'Position', [360 20 720 280]);
    title(axVoltage, 'DC Bus Voltage Regulation (Nominal 600V)');
    xlabel(axVoltage, 'Simulation Step'); ylabel(axVoltage, 'Voltage (V)');
    grid(axVoltage, 'on'); hold(axVoltage, 'on');
    yline(axVoltage, 600, 'r--', 'LineWidth', 2);
    ylim(axVoltage, [0 700]);
    
    lineDemand = animatedline(axPower, 'Color', 'k', 'LineWidth', 2, 'DisplayName', 'Demand (kW)');
    lineTotalGen = animatedline(axPower, 'Color', '#A2142F', 'LineWidth', 2, 'LineStyle', '-.', 'DisplayName', 'Total Gen (kW)');
    linePV = animatedline(axPower, 'Color', '#D95319', 'LineWidth', 1.5, 'DisplayName', 'PV');
    lineFC = animatedline(axPower, 'Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'FC');
    lineWT = animatedline(axPower, 'Color', '#77AC30', 'LineWidth', 1.5, 'DisplayName', 'WT');
    legend(axPower, 'Location', 'northeast');
    lineVoltage = animatedline(axVoltage, 'Color', '#7E2F8E', 'LineWidth', 2);
    
    tab2 = uitab(tg, 'Title', 'AI Scheduler Benchmarking');
    axRMSE = uiaxes(tab2, 'Position', [20 100 340 450]);
    bar(axRMSE, categorical(modelNames), rmseList, 'FaceColor', '#0072BD');
    title(axRMSE, 'Model Accuracy (RMSE)');
    ylabel(axRMSE, 'Prediction Error (Lower is Better)', 'FontWeight', 'bold');
    grid(axRMSE, 'on');
    axAccuracy = uiaxes(tab2, 'Position', [385 100 340 450]);
    bar(axAccuracy, categorical(modelNames), accuracyList, 'FaceColor', '#77AC30');
    title(axAccuracy, 'Model Accuracy (%)');
    ylabel(axAccuracy, 'Accuracy % (Higher is Better)', 'FontWeight', 'bold');
    ylim(axAccuracy, [0 100]);
    grid(axAccuracy, 'on');
    axLatency = uiaxes(tab2, 'Position', [750 100 340 450]);
    bar(axLatency, categorical(modelNames), latencyList, 'FaceColor', '#D95319');
    title(axLatency, 'Inference Speed (Latency)');
    ylabel(axLatency, 'Time per step [ms] (Lower is Better)', 'FontWeight', 'bold');
    grid(axLatency, 'on');
    uilabel(tab2, 'Text', sprintf('Selected model: %s   |   Accuracy: %.2f%%   |   R^2: %.4f   |   RMSE: %.5f   |   Latency: %.3f ms/step', ...
        bestModelName, bestAccuracy, bestR2, rmseList(bestIdx), bestLatency), ...
        'Position', [20 30 1070 40], 'FontSize', 13, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        
    tab3 = uitab(tg, 'Title', 'Live Sensor & Signal Monitor');
    sensorSpecs = { ...
        'Solar_Irradiance',  'Solar Irradiance (W/m^2)',        '#EDB120'; ...
        'Wind_Speed',        'Wind Speed (m/s)',                '#4DBEEE'; ...
        'Load_Demand',       'Load Demand (kW)',                'k'; ...
        'Temperature',       'Ambient Temperature (deg C)',     '#D95319'; ...
        'Relative_Humidity', 'Relative Humidity (%)',           '#0072BD'; ...
        'Surface_Pressure',  'Surface Pressure (kPa)',          '#7E2F8E'; ...
        'Battery_SOC',       'Battery SOC (%)',                 '#77AC30'; ...
        'Time_Index',        'Time Index (Hour of Day)',        '#A2142F'; ...
        'Season_Index',      'Season Index (sin, annual)',      '#4DBEEE'; ...
        'Power_Imbalance',   'Power Imbalance (kW)',            '#D95319'; ...
        'Previous_Action',   'Previous Action (mean alpha)',    '#0072BD' ...
    };
    numSensors = size(sensorSpecs, 1);
    numCols = 4; numRows = ceil(numSensors / numCols);
    cellW = 258; cellH = 190; gap = 10; originX = 15; originY = 20;
    sensorAxes = gobjects(1, numSensors);
    sensorLines = gobjects(1, numSensors);
    for k = 1:numSensors
        col = mod(k-1, numCols);
        row = floor((k-1) / numCols);
        xPos = originX + col * (cellW + gap);
        yPos = originY + (numRows - 1 - row) * (cellH + gap);
        ax = uiaxes(tab3, 'Position', [xPos, yPos, cellW, cellH]);
        title(ax, sensorSpecs{k,2}, 'FontSize', 9);
        xlabel(ax, 'Step', 'FontSize', 8); 
        grid(ax, 'on'); hold(ax, 'on');
        ax.FontSize = 7;
        sensorAxes(k) = ax;
        sensorLines(k) = animatedline(ax, 'Color', sensorSpecs{k,3}, 'LineWidth', 1.3);
    end
    disp('Live Sensor & Signal Monitor');
    
    logInputs = zeros(N, numSensors);
    logAIFraction = zeros(N, 3);
    logAIGenKW = zeros(N, 4);
    logBattery = zeros(N, 3);
    logMismatchFinal = zeros(N, 1);
    logVoltage = zeros(N, 1);
    logGridMode = strings(N, 1);
    stepsCompleted = 0;
    disp('Simulation actively streaming in Web UI Tab 1 and Tab 3...');
    sim_SoC = 0.5;
    prevAlpha = 0;
    
    for t = 1:N
        if ~isvalid(appFig), break; end
        currentFeatures = X_matrix(:, t);
        currentFeatures(7) = sim_SoC;
        currentDemand_kW = raw_Demand(t) / 1000;
        alpha_pred = predict(bestNet, currentFeatures);
        
        pv_gen = alpha_pred(1) * 80;
        fc_gen = alpha_pred(2) * 50;
        wt_gen = alpha_pred(3) * 30;
        total_gen = pv_gen + fc_gen + wt_gen;
        
        imbalance_kW = total_gen - currentDemand_kW;
        mismatch_kW = currentDemand_kW - total_gen;
        
        bat_charge = 0; bat_discharge = 0;
        if mismatch_kW > 0
            if sim_SoC > 0.2
                bat_discharge = min(mismatch_kW, 40);
                mismatch_kW = mismatch_kW - bat_discharge;
                sim_SoC = sim_SoC - (bat_discharge * 1e-3);
            end
        else
            if sim_SoC < 0.9
                bat_charge = min(abs(mismatch_kW), 40);
                mismatch_kW = mismatch_kW + bat_charge;
                sim_SoC = sim_SoC + (bat_charge * 1e-3);
            end
        end
        
        gridDeadband = 0.05;
        if mismatch_kW > gridDeadband
            gridModeStr = "IMPORTING";
            lblGridState.Text = 'GRID MODE: IMPORTING'; lblGridState.BackgroundColor = '#D95319';
        elseif mismatch_kW < -gridDeadband
            gridModeStr = "EXPORTING";
            lblGridState.Text = 'GRID MODE: EXPORTING'; lblGridState.BackgroundColor = '#77AC30';
        else
            gridModeStr = "ISLANDED";
            lblGridState.Text = 'GRID MODE: ISLANDED'; lblGridState.BackgroundColor = '#0072BD';
        end
        
        lblLoad.Text = sprintf('Tongi Area Load: %.1f kW', currentDemand_kW);
        lblTotalGen.Text = sprintf('Total Generation: %.1f kW', total_gen);
        
        if pv_gen > 0.5, lblPV.Text = sprintf('PV Energy: %.1f kW (GENERATING)', pv_gen); lblPV.FontColor = '#D95319'; else lblPV.Text = sprintf('PV Energy: %.1f kW (STANDBY)', pv_gen); lblPV.FontColor = 'k'; end
        if wt_gen > 0.5, lblWT.Text = sprintf('Wind Turbine: %.1f kW (GENERATING)', wt_gen); lblWT.FontColor = '#77AC30'; else lblWT.Text = sprintf('Wind Turbine: %.1f kW (STANDBY)', wt_gen); lblWT.FontColor = 'k'; end
        if fc_gen > 0.5, lblFC.Text = sprintf('Fuel Energy: %.1f kW (GENERATING)', fc_gen); lblFC.FontColor = '#0072BD'; else lblFC.Text = sprintf('Fuel Energy: %.1f kW (STANDBY)', fc_gen); lblFC.FontColor = 'k'; end
        
        battDeadband = 0.05;
        if bat_charge > battDeadband
            lblBESS.Text = sprintf('Battery SOC: %.1f%% (CHARGING, +%.2f kW)', sim_SoC * 100, bat_charge); lblBESS.FontColor = '#77AC30';
            lblDCDC.Text = 'DC-DC Strategy: ABSORBING SURPLUS'; lblDCDC.FontColor = '#77AC30';
        elseif bat_discharge > battDeadband
            lblBESS.Text = sprintf('Battery SOC: %.1f%% (DISCHARGING, -%.2f kW)', sim_SoC * 100, bat_discharge); lblBESS.FontColor = '#D95319';
            lblDCDC.Text = 'DC-DC Strategy: SUPPLYING DEFICIT'; lblDCDC.FontColor = '#D95319';
        else
            lblBESS.Text = sprintf('Battery SOC: %.1f%% (IDLE)', sim_SoC * 100); lblBESS.FontColor = 'k';
            lblDCDC.Text = 'DC-DC Strategy: IDLE'; lblDCDC.FontColor = 'k';
        end
        
        voltage_deviation = (total_gen - currentDemand_kW) / currentDemand_kW;
        bus_voltage = 600 + (voltage_deviation * 20) + (randn * 1.2);
        
        addpoints(lineDemand, t, currentDemand_kW);
        addpoints(lineTotalGen, t, total_gen);
        addpoints(linePV, t, pv_gen);
        addpoints(lineFC, t, fc_gen);
        addpoints(lineWT, t, wt_gen);
        addpoints(lineVoltage, t, bus_voltage);
        
        seasonVal = sin(2*pi*t/8760);
        sensorValues = [raw_G(t), raw_W(t), currentDemand_kW, raw_T(t), raw_RH(t), ...
                         raw_PS(t), sim_SoC*100, mod(t,24), seasonVal, imbalance_kW, prevAlpha];
        for k = 1:numSensors
            addpoints(sensorLines(k), t, sensorValues(k));
        end
        
        logInputs(t, :) = sensorValues;
        logAIFraction(t, :) = alpha_pred(:)';
        logAIGenKW(t, :) = [pv_gen, fc_gen, wt_gen, total_gen];
        logBattery(t, :) = [bat_charge, bat_discharge, sim_SoC * 100];
        logMismatchFinal(t) = mismatch_kW;
        logVoltage(t) = bus_voltage;
        logGridMode(t) = gridModeStr;
        stepsCompleted = t;
        prevAlpha = mean(alpha_pred);
        
        if t > 100
            xlim(axPower, [t-100, t+10]); xlim(axVoltage, [t-100, t+10]);
            for k = 1:numSensors
                xlim(sensorAxes(k), [t-100, t+10]);
            end
        else
            xlim(axPower, [0, 110]); xlim(axVoltage, [0, 110]);
            for k = 1:numSensors
                xlim(sensorAxes(k), [0, 110]);
            end
        end
        drawnow limitrate;
        pause(0.005);
    end
    
    disp('--- Autonomous Platform Execution Complete ---');
    if stepsCompleted > 0
        rows = 1:stepsCompleted;
        inputColNames = {'Solar_Irradiance_Wm2', 'Wind_Speed_ms', 'Load_Demand_kW', 'Temperature_C', ...
                          'Relative_Humidity_pct', 'Surface_Pressure_kPa', 'Battery_SOC_Input_pct', ...
                          'Time_Index_Hour', 'Season_Index', 'Power_Imbalance_Input_kW', 'Previous_Action'};
        simLogTable = table((rows)', 'VariableNames', {'Step'});
        for k = 1:numSensors
            simLogTable.(inputColNames{k}) = logInputs(rows, k);
        end
        simLogTable.AI_PV_Fraction        = logAIFraction(rows, 1);
        simLogTable.AI_FC_Fraction        = logAIFraction(rows, 2);
        simLogTable.AI_WT_Fraction        = logAIFraction(rows, 3);
        simLogTable.PV_Generation_kW      = logAIGenKW(rows, 1);
        simLogTable.FC_Generation_kW      = logAIGenKW(rows, 2);
        simLogTable.WT_Generation_kW      = logAIGenKW(rows, 3);
        simLogTable.Total_Generation_kW   = logAIGenKW(rows, 4);
        simLogTable.Battery_Charge_kW     = logBattery(rows, 1);
        simLogTable.Battery_Discharge_kW  = logBattery(rows, 2);
        simLogTable.Battery_SOC_Final_pct = logBattery(rows, 3);
        simLogTable.Power_Mismatch_Final_kW = logMismatchFinal(rows);
        simLogTable.Grid_Mode             = logGridMode(rows);
        simLogTable.DC_Bus_Voltage_V      = logVoltage(rows);
        
        simLogCsvName = sprintf('Microgrid_Simulation_Log_%s.csv', runTimestamp);
        try
            writetable(simLogTable, simLogCsvName);
            fprintf('Full simulation log (AI inputs + final outputs, %d steps) exported to: %s\n', ...
                stepsCompleted, simLogCsvName);
        catch ME
            warning('Could not write simulation log CSV (%s).', ME.message);
        end
    else
        disp('No simulation steps completed - skipping CSV export.');
    end
end