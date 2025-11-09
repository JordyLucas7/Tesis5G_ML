% =========================================================================
% SIMULACIÓN DE RED 5G CONGESTIÓN Y OPTIMIZACIÓN (VERSIÓN REGRESIÓN LOGÍSTICA)
%
% BASADO EN EL SCRIPT FUNDAMENTAL (v5)
% 1. [MODELO] Enrutador base reemplazado por Regresión Logística.
% 2. [ARREGLO DE ERROR v7] Corregido el error 'Number of observations...
%    disagree'. 'trainNetwork' ahora recibe 'X' como [Samples x Features]
%    y 'Y' como un vector categórico [Samples x 1].
% 3. [MÉTRICA] Se usa Validación 80/20 para calcular Error (Fórmula 2)
%    y Eficacia (Fórmula 3).
% 4. [LÓGICA GRADUAL] Se mantiene la lógica de optimización gradual.
% 5. [rng(1)] Se mantiene para datos deterministas.
% 6. [RUTA] Guardado en carpeta LOGR (Regresión Logística).
% =========================================================================

clear; clc; close all;

% --- Fija la semilla del generador aleatorio (DATOS CERTEROS) ---
rng(1, 'twister');

%% ===== PARAMETROS =====
numRoutersInit = 15;
numRoutersMax = 30; % Total final de routers
maxDevicesPerRouter = 70; % congestión máxima inicial
simTimeCong = 60;
simTimeOpt = 30;
simTimeTot = simTimeCong + simTimeOpt;
routersAll = strcat("Router", string(1:numRoutersMax));

% --- Parámetros para introducción gradual ---
routersToAdd = numRoutersMax - numRoutersInit; % 15 routers nuevos
addInterval = 5; % Añadir routers cada 5 segundos
routersPerInterval = 3; % Añadir 3 routers por intervalo
numIntervals = routersToAdd / routersPerInterval; % 5 intervalos
% --- Fin parámetros graduales ---

% Colores dispositivos
pcColor = [0 0.6 0]; % Verde para PCs
phoneColor = [1 0 0]; % Rojo para Móviles

%% ===== CLASIFICADOR DE COLOR (Árbol de Decisión Simple) ====
% Entrenado para clasificar según carga (usado solo para visualización)
trainDataColor = []; labelsColor = [];
critThresholdColor = 46; % >= 46 Crítico (Rojo)
alertThresholdColor = 26; % 26-45 Alerta (Naranja)
% <= 25 Normal (Verde)
for d = 1:maxDevicesPerRouter
    if d >= critThresholdColor
        stateColor = "Critico";
    elseif d >= alertThresholdColor
        stateColor = "Alerta";
    else
        stateColor = "Normal";
    end
    trainDataColor = [trainDataColor; d];
    labelsColor = [labelsColor; stateColor];
end
labelsColor = categorical(labelsColor);
Mdl_Color = fitctree(trainDataColor, labelsColor); % Árbol simple es suficiente


%% ===== REGRESIÓN LOGÍSTICA (Entrenamiento con Deep Learning Toolbox) ====
% --- INICIO DE MODIFICACIÓN: ANN -> Regresión Logística ---
disp('Preparando datos de entrenamiento para Regresión Logística (Enrutador)...');
datasetLOGR = []; labelsLOGR = []; % Labels para decisión de carga
critThresholdLOGR = 46; % Umbrales para el OBJETIVO del enrutador
alertThresholdLOGR = 26;
for d = 10:1:maxDevicesPerRouter
     for i = 1:10
        lat = 10 + (d^1.3)/10 + rand*3;
        loss = min(40, (d/3) + rand*3);
        thr = max(0.5, 9 - (d^1.1)/20 + rand*0.3);

        if d >= critThresholdLOGR
            targetStateLOGR = "Critico";
        elseif d >= alertThresholdLOGR
            targetStateLOGR = "Alerta";
        else
            targetStateLOGR = "Normal";
        end
        % Features: [dispositivos, throughput, latencia, perdida]
        datasetLOGR = [datasetLOGR; d thr lat loss];
        labelsLOGR = [labelsLOGR; targetStateLOGR];
    end
end

% Preparar datos para trainNetwork (Validación 80/20)
X_matrix = datasetLOGR;
Y_cat = categorical(labelsLOGR);
classNamesLOGR = categories(Y_cat); % Guardar nombres de clases

numObservaciones = size(X_matrix, 1);
cv = cvpartition(numObservaciones, 'HoldOut', 0.2);
idxTrain = training(cv);
idxValidation = test(cv);

% --- INICIO DE CORRECCIÓN DE ERROR v7 ---
% Formato para trainNetwork [Samples x Features]
XTrain = X_matrix(idxTrain, :); % [N_train x 4] (SIN transponer)
YTrain_cat = Y_cat(idxTrain); % [N_train x 1] Categórico
XValidation = X_matrix(idxValidation, :); % [N_val x 4] (SIN transponer)
YValidation_cat = Y_cat(idxValidation); % [N_val x 1] Categórico
% --- FIN DE CORRECCIÓN DE ERROR ---

% Definir Arquitectura (Regresión Logística = Red Neuronal simple)
numFeatures = 4; % d, thr, lat, loss
numClasses = numel(classNamesLOGR);
layers = [ 
    featureInputLayer(numFeatures)
    fullyConnectedLayer(numClasses) % Conexión directa de entrada a salida
    softmaxLayer
    classificationLayer];

% Opciones de entrenamiento
% --- INICIO DE CORRECCIÓN DE ERROR v7 ---
% ValidationData ahora usa XValidation [N_val x 4] y YValidation_cat [N_val x 1]
options = trainingOptions('adam', 'ExecutionEnvironment', 'auto', 'MaxEpochs', 100, ...
    'MiniBatchSize', 32, 'ValidationData', {XValidation, YValidation_cat}, ...
    'GradientThreshold', 1, 'Verbose', false, 'Plots', 'none');
% --- FIN DE CORRECCIÓN DE ERROR ---

% Entrenamiento del enrutador (Regresión Logística Multinomial)
disp('Entrenando modelo de Regresión Logística (Enrutador) con 80% de datos...');
% --- INICIO DE CORRECCIÓN DE ERROR v7 ---
% trainNetwork ahora usa XTrain [N_train x 4] y YTrain_cat [N_train x 1]
[Mdl_LOGR, trainInfo] = trainNetwork(XTrain, YTrain_cat, layers, options);
% --- FIN DE CORRECCIÓN DE ERROR ---

% Calcular métrica de eficiencia (Precisión de Validación)
YValidation_pred_cat = classify(Mdl_LOGR, XValidation); % classify SÍ acepta [Samples x Features]

logrAccuracyPercent = 100 * sum(YValidation_pred_cat == YValidation_cat) / numel(YValidation_cat); % Eficacia (Fórmula 3)
logrErrorPercent = 100 - logrAccuracyPercent; % Error (Fórmula 2)
disp(['Eficacia de validación Reg. Logística (Fórmula 3): ', num2str(logrAccuracyPercent, '%.2f'), '%']);
disp(['Error de validación Reg. Logística (Fórmula 2): ', num2str(logrErrorPercent, '%.2f'), '%']);
% --- FIN DE MODIFICACIÓN ---


%% ===== INICIALIZAR VARIABLES =====
latency = NaN(simTimeTot, numRoutersMax); % Usar NaN para routers inactivos
throughput = NaN(simTimeTot, numRoutersMax);
lostPackets = NaN(simTimeTot, numRoutersMax);
energyConsumption = NaN(simTimeTot, numRoutersMax);
packetsSent = zeros(simTimeTot, numRoutersMax); % Usar 0 para inactivos
packetsReceived = zeros(simTimeTot, numRoutersMax);
devicesConnected = zeros(simTimeTot, numRoutersMax);
routerStatesColor = strings(simTimeTot, numRoutersMax); % Para colores
routerStatesColor(:,:) = "Inactivo"; % Inicializar como inactivo
finalPCs = zeros(numRoutersMax, 1);
finalPhones = zeros(numRoutersMax, 1);

%% ===== CONFIGURACIÓN DE GUARDADO (LOGR) =====
% --- RUTA ACTUALIZADA ---
savePath = 'C:\Users\jordy\Documents\Resultados\LOGR\';
% --- FIN ACTUALIZACIÓN ---
if ~exist(savePath, 'dir')
    mkdir(savePath);
end
disp('--- INICIANDO SIMULACIÓN REGRESIÓN LOGÍSTICA ---');

%% ===== FIGURA PRINCIPAL =====
fig1 = figure('Name', 'Simulación Red 5G - Congestión y Optimización (Reg. Logística)', 'Position', [50 50 1600 800]);

%% ===== SIMULACIÓN =====
tic; % medir tiempo de enrutamiento
routersActivos = numRoutersInit; % Empezar con 15

for t = 1:simTimeTot

    if t > simTimeCong && mod(t - simTimeCong - 1, addInterval) == 0 && routersActivos < numRoutersMax
        routersNuevosEsteIntervalo = min(routersPerInterval, numRoutersMax - routersActivos);
        routersActivos = routersActivos + routersNuevosEsteIntervalo;
        disp(['t=', num2str(t), 's: Activando ', num2str(routersNuevosEsteIntervalo), ' routers nuevos. Total activos: ', num2str(routersActivos)]);
    end

    Gtemp = graph();
    activeRouterNames = routersAll(1:routersActivos);
    Gtemp = addnode(Gtemp, activeRouterNames);
    for i = 1:routersActivos
        for j = i+1:routersActivos
            Gtemp = addedge(Gtemp, activeRouterNames(i), activeRouterNames(j));
        end
    end

    for r = 1:routersActivos

        if t <= simTimeCong
             if r <= numRoutersInit
                numTotal = round((t/simTimeCong) * maxDevicesPerRouter);
             else
                 numTotal = 0;
             end
        else
            isNewRouter = (r > numRoutersInit);
            prevStateTime = max(1, t-1); 

            % --- INICIO DE MODIFICACIÓN: ANN -> Regresión Logística ---
            if isNewRouter && devicesConnected(prevStateTime, r) == 0
                 features_logr_raw = [0, 10, 5, 0]; % Estado ideal [1 x 4]
                 predLOGR_Enrutador = "Normal"; 
            else
                 prevDev = devicesConnected(prevStateTime,r);
                 prevThr = fillmissing(throughput(prevStateTime,r),'constant', 10); 
                 prevLat = fillmissing(latency(prevStateTime,r),'constant', 5); 
                 prevLoss = fillmissing(lostPackets(prevStateTime,r),'constant', 0); 
                 features_logr_raw = [prevDev, prevThr, prevLat, prevLoss]; % Formato [1 x 4]

                 % --- INICIO DE CORRECCIÓN DE ERROR v7 ---
                 % 'classify' espera [Samples x Features] (o sea, [1 x 4])
                 predLOGR_Enrutador = classify(Mdl_LOGR, features_logr_raw);
                 % --- FIN DE CORRECCIÓN DE ERROR v7 ---
            end
            % --- FIN DE MODIFICACIÓN ---

            % Política de enrutamiento GRADUAL (NUEVO ESTÁNDAR)
            if strcmp(predLOGR_Enrutador, 'Critico')
                numTotal = randi([10, 20]); 
            elseif strcmp(predLOGR_Enrutador, 'Alerta')
                numTotal = randi([20, 30]); 
            else % "Normal"
                numTotal = randi([30, 45]); 
            end

             if isNewRouter && devicesConnected(prevStateTime, r) == 0 
                  numTotal = randi([5, 10]);
             elseif numTotal <= 0 && devicesConnected(prevStateTime,r) > 0 && ~isNewRouter
                 numTotal = randi([1,5]);
             end
        end
        devicesConnected(t,r) = numTotal;

        numPCs = randi([round(0.4*numTotal), round(0.6*numTotal)]);
        numPhones = numTotal - numPCs;
        if numPCs > 0; pcs = strcat("PC", string(r), "_", string(1:numPCs)); Gtemp = addnode(Gtemp, pcs); for L = 1:numPCs; Gtemp = addedge(Gtemp, activeRouterNames(r), pcs(L)); end; else; pcs = strings(0); end
        if numPhones > 0; phones = strcat("Movil", string(r), "_", string(1:numPhones)); Gtemp = addnode(Gtemp, phones); for P = 1:numPhones; Gtemp = addedge(Gtemp, activeRouterNames(r), phones(P)); end; else; phones = strings(0); end

        if t == simTimeTot
            finalPCs(r) = numPCs;
            finalPhones(r) = numPhones;
        end

        if numTotal > 0; trafficPCs = poissrnd(60, [1 numPCs]); trafficPhones = poissrnd(90, [1 numPhones]) + randn(1,numPhones)*15; trafficPhones(trafficPhones < 0) = 0; currentTraffic = sum(trafficPCs) + sum(trafficPhones); packetsSent(t,r) = currentTraffic / 10;
        else; currentTraffic = 0; packetsSent(t,r) = 0; end

        if t <= simTimeCong; if numTotal > 0; packetsReceived(t,r) = packetsSent(t,r) * (0.8 - 0.3*(numTotal/maxDevicesPerRouter)); else; packetsReceived(t,r) = 0; end
        else; step = (t - simTimeCong) / simTimeOpt; if numTotal > 0; packetsReceived(t,r) = packetsSent(t,r) * (0.5 + 0.5*step); else; packetsReceived(t,r) = 0; end; end

        if numTotal > 0
            latency(t,r) = max(5, 7 + (numTotal^1.2)/12 + rand*1.5);
            lostPackets(t,r) = max(0, min(25, (numTotal/3.5) + rand*1.5));
            throughput(t,r) = max(1, 9 - (numTotal^1.0)/25 + rand*0.2);
            energyConsumption(t,r) = max(0.4, 0.7 + 0.01*numTotal + rand*0.1);
        else; latency(t,r) = NaN; lostPackets(t,r) = NaN; throughput(t,r) = NaN; energyConsumption(t,r) = NaN; end
    end % Fin bucle for r

    for r = 1:routersActivos
        predColor = predict(Mdl_Color, devicesConnected(t,r));
        routerStatesColor(t,r) = string(predColor);
    end

    % ===== VISUALIZACIÓN =====
    subplot(1,2,1); cla; h = plot(Gtemp, 'Layout', 'force'); h.NodeLabel = {};
    nodeIndicesPC = findnode(Gtemp, Gtemp.Nodes.Name(contains(Gtemp.Nodes.Name, "PC")));
    if ~isempty(nodeIndicesPC); highlight(h, nodeIndicesPC, 'NodeColor', pcColor, 'MarkerSize', 3); end
    nodeIndicesMovil = findnode(Gtemp, Gtemp.Nodes.Name(contains(Gtemp.Nodes.Name, "Movil")));
    if ~isempty(nodeIndicesMovil); highlight(h, nodeIndicesMovil, 'NodeColor', phoneColor, 'MarkerSize', 3); end
    for r = 1:routersActivos
        state = routerStatesColor(t,r); nodeIdx = findnode(Gtemp, activeRouterNames(r));
         if nodeIdx > 0; if state == "Normal"; color = [0 0.7 0]; elseif state == "Alerta"; color = [1 0.7 0]; else; color = [0.85 0.1 0.1]; end
            highlight(h, nodeIdx, 'NodeColor', color, 'MarkerSize', 9); end
    end
    if t <= simTimeCong; title(sprintf('Congestión - Segundo %d / %d (%d Routers)', t, simTimeTot, routersActivos));
    else; title(sprintf('Optimización (Reg. Logística) - Segundo %d / %d (%d Routers)', t, simTimeTot, routersActivos)); end

    subplot(2,2,2); cla; plot(1:t, mean(latency(1:t,1:routersActivos), 2,'omitnan'), 'r', 'LineWidth', 1.5); xline(60, '--k', 'Inicio optimización'); title('Latencia promedio (En vivo)'); xlabel('Tiempo (s)'); ylabel('ms'); grid on; xlim([0 simTimeTot]); ylim('auto');
    subplot(2,2,4); cla; plot(1:t, mean(throughput(1:t,1:routersActivos), 2,'omitnan'), 'b', 'LineWidth', 1.5); xline(60,'--k', 'Inicio optimización'); title('Throughput promedio (En vivo)'); xlabel('Tiempo (s)'); ylabel('Gbps'); grid on; xlim([0 simTimeTot]); ylim('auto');

    drawnow; pause(0.05);
end % Fin bucle for t

elapsedTime = toc*1000;
disp(['Tiempo total de simulación: ', num2str(elapsedTime/1000, '%.2f'), ' segundos']);

%% ===== RESULTADOS =====
avgLatencyBefore = mean(mean(latency(1:simTimeCong,1:numRoutersInit),'omitnan'));
avgLatencyAfter = mean(mean(latency(simTimeCong+1:end, 1:numRoutersMax),'omitnan'));
avgThrBefore = mean(mean(throughput(1:simTimeCong, 1:numRoutersInit),'omitnan'));
avgThrAfter = mean(mean(throughput(simTimeCong+1:end, 1:numRoutersMax),'omitnan'));
avgLossBefore = mean(mean(lostPackets(1:simTimeCong, 1:numRoutersInit),'omitnan'));
avgLossAfter = mean(mean(lostPackets(simTimeCong+1:end, 1:numRoutersMax),'omitnan'));
avgEnergyBefore = mean(mean(energyConsumption(1:simTimeCong, 1:numRoutersInit),'omitnan'));
avgEnergyAfter = mean(mean(energyConsumption(simTimeCong+1:end, 1:numRoutersMax),'omitnan'));
resultsBefore = table(avgLatencyBefore, avgThrBefore, avgLossBefore, avgEnergyBefore, 'VariableNames', {'Latencia_ms', 'Throughput_Gbps', 'Perdida_p', 'Energia_mJ'});
resultsAfter = table(avgLatencyAfter, avgThrAfter, avgLossAfter, avgEnergyAfter, 'VariableNames', {'Latencia_ms', 'Throughput_Gbps', 'Perdida_p', 'Energia_mJ'});
improvement = ((resultsBefore{:,:} - resultsAfter{:,:}) ./ resultsBefore{:,:}) * 100;
improvement(2) = ((resultsAfter{:,2} - resultsBefore{:,2}) / resultsBefore{:,2}) * 100; 
resultsComparison = array2table(improvement, 'VariableNames', {'Mejora_Latencia_p', 'Mejora_Throughput_p', 'Mejora_Perdida_p', 'Mejora_Energia_p'});

disp('--- RESULTADOS ANTES DE OPTIMIZACIÓN (Fase Congestión t=1-60) ---'); disp(resultsBefore);
disp('--- RESULTADOS DESPUÉS DE OPTIMIZACIÓN (Fase Reg. Logística t=61-90) ---'); disp(resultsAfter);
disp('--- % DE MEJORA ENTRE CONGESTIÓN Y OPTIMIZACIÓN ---'); disp(resultsComparison);

%% ===== TABLA DETALLADA POR ROUTER =====
routerNames = routersAll(1:numRoutersMax)';
finalDevices = devicesConnected(end,1:numRoutersMax)';
finalSent = packetsSent(end,1:numRoutersMax)'/1e6;
finalReceived = packetsReceived(end,1:numRoutersMax)'/1e6;
finalEnergy = energyConsumption(end,1:numRoutersMax)';
finalLost = lostPackets(end,1:numRoutersMax)';
finalStateColor = routerStatesColor(end,1:numRoutersMax)';
routerDetails = table(routerNames, finalDevices, finalPCs(1:numRoutersMax), finalPhones(1:numRoutersMax), finalSent, finalReceived, finalLost, finalEnergy, finalStateColor, 'VariableNames', {'Router', 'Dispositivos', 'PCs', 'Moviles', 'Paquetes_Enviados_M', 'Paquetes_Recibidos_M', 'Perdida_p', 'Consumo_Energia_mJ', 'Estado_Final_Color'});
disp('--- DETALLES POR ROUTER (Estado Final t=90s) ---'); disp(routerDetails);


%% ===== MÉTRICAS DE EFICIENCIA DEL LOGR (Fórmulas 2 y 3) =====
timeToOptimize = simTimeOpt;
latencyReduction = improvement(1);
lossReduction = improvement(3);

avgEnergyJoule = avgEnergyAfter / 1000; % (mJ/s) -> (J/s)
avgBitsPerSec = avgThrAfter * 1e9;    % (Gbit/s) -> (bit/s)
if avgThrAfter > 0 && avgEnergyJoule > 0; effEnergy_J_per_bit = avgEnergyJoule / avgBitsPerSec; else; effEnergy_J_per_bit = 0; end
effEnergy_pJ_per_bit = effEnergy_J_per_bit * 1e12; % Convertir J/bit a pJ/bit

logrAccuracyPercent = logrAccuracyPercent; % Tu Fórmula 3 (Eficacia)
logrErrorPercent = logrErrorPercent; % Tu Fórmula 2 (Error)

efficiencyMetrics = table(elapsedTime, timeToOptimize, latencyReduction, lossReduction, effEnergy_pJ_per_bit, logrErrorPercent, logrAccuracyPercent, ...
    'VariableNames', {'TiempoEnrutamiento_ms', 'TiempoOptimiz_s', 'ReduccionLatencia_p', 'ReduccionPerdida_p', 'EficienciaEnergetica_pJbit', 'ErrorValidacion_LOGR_p', 'EficaciaValidacion_LOGR_p'});
disp('--- MÉTRICAS DE EFICIENCIA DE REG. LOGÍSTICA (Enrutador) ---'); disp(efficiencyMetrics);


%% ===== GUARDAR RESULTADOS DE REGRESIÓN LOGÍSTICA =====
Resultados_LOGR.Tablas.Before = resultsBefore;
Resultados_LOGR.Tablas.After = resultsAfter;
Resultados_LOGR.Tablas.Comparison = resultsComparison;
Resultados_LOGR.Tablas.RouterDetails = routerDetails;
Resultados_LOGR.Tablas.Efficiency = efficiencyMetrics;
Resultados_LOGR.Series.Latency = latency;
Resultados_LOGR.Series.Throughput = throughput;
Resultados_LOGR.Series.Loss = lostPackets;
Resultados_LOGR.Series.Energy = energyConsumption;
Resultados_LOGR.Series.StatesColor = routerStatesColor;
Resultados_LOGR.Modelo.LOGR_Model = Mdl_LOGR;
Resultados_LOGR.Modelo.TrainInfo = trainInfo;

save(fullfile(savePath, 'Resultados_LOGR.mat'), 'Resultados_LOGR');
writetable(resultsBefore, fullfile(savePath, 'Resultados_LOGR_Before.csv'));
writetable(resultsAfter, fullfile(savePath, 'Resultados_LOGR_After.csv'));
writetable(resultsComparison, fullfile(savePath, 'Resultados_LOGR_Comparison.csv'));
writetable(routerDetails, fullfile(savePath, 'Resultados_LOGR_RouterDetails.csv'));
writetable(efficiencyMetrics, fullfile(savePath, 'Resultados_LOGR_Efficiency.csv'));

disp(['--- TODOS LOS RESULTADOS DE REG. LOGÍSTICA SE HAN GUARDADO EN: ', savePath, ' ---']);

%% ===== GRÁFICOS COMPARATIVOS (FIGURAS 2-5) =====

% La figura 1 (en vivo) se deja abierta
fig2 = figure('Name', 'Comparación Antes vs Después', 'Position', [100 100 1200 600], 'Visible', 'off');
subplot(2,2,1); b1 = bar([avgLatencyBefore avgLatencyAfter], 'r'); ylabel('ms'); title('Latencia promedio (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgLatencyBefore, avgLatencyAfter, 1])*1.2]); text(b1.XEndPoints, b1.YEndPoints, sprintfc('%.2f ms', b1.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
subplot(2,2,2); b2 = bar([avgThrBefore avgThrAfter], 'b'); ylabel('Gbps'); title('Throughput promedio (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgThrBefore, avgThrAfter, 1])*1.2]); text(b2.XEndPoints, b2.YEndPoints, sprintfc('%.2f Gbps', b2.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
subplot(2,2,3); b3 = bar([avgLossBefore avgLossAfter], 'FaceColor', [1 0.5 0]); ylabel('%'); title('Pérdida de paquetes (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgLossBefore, avgLossAfter, 1])*1.2]); text(b3.XEndPoints, b3.YEndPoints, sprintfc('%.2f %%', b3.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
subplot(2,2,4); b4 = bar([avgEnergyBefore avgEnergyAfter], 'g'); ylabel('mJ'); title('Consumo energético (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgEnergyBefore, avgEnergyAfter, 0.1])*1.2]); text(b4.XEndPoints, b4.YEndPoints, sprintfc('%.2f mJ', b4.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
drawnow; saveas(fig2, fullfile(savePath, 'LOGR_ComparacionAntesDespues.png')); close(fig2);


fig3 = figure('Name', 'Topologia Final', 'Position', [100 100 900 600], 'Visible', 'off');
Gfinal = graph(); finalActiveRouters = routersAll(1:routersActivos); Gfinal = addnode(Gfinal, finalActiveRouters);
for i = 1:routersActivos; for j = i+1:routersActivos; Gfinal = addedge(Gfinal, finalActiveRouters(i), finalActiveRouters(j)); end; end
hFinal = plot(Gfinal, 'Layout', 'force', 'NodeLabel', {});
title('Topologia final - Estado Inicial (t=Congestión) vs Final (t=Optimización)');
colorsFinal = zeros(routersActivos, 3);
for r = 1:routersActivos
    routerName = finalActiveRouters(r); routerIndexGlobal = find(strcmp(routersAll, routerName));
    finalState = routerStatesColor(end, routerIndexGlobal);
    if finalState == "Normal"; colorsFinal(r,:) = [0 0.7 0]; elseif finalState == "Alerta"; colorsFinal(r,:) = [1 0.7 0]; else; colorsFinal(r,:) = [0.85 0.1 0.1]; end
    nodeIdxInGfinal = findnode(Gfinal, routerName);
    if nodeIdxInGfinal > 0
        highlight(hFinal, nodeIdxInGfinal, 'NodeColor', colorsFinal(r,:), 'MarkerSize', 8);
        if routerIndexGlobal <= numRoutersInit; textInicial = sprintf('t60:%s(%dd)', routerStatesColor(simTimeCong, routerIndexGlobal), devicesConnected(simTimeCong, routerIndexGlobal));
        else; textInicial = 'tc:Nuevo'; end
        textFinal = sprintf('to:%s(%dd)', routerStatesColor(end, routerIndexGlobal), devicesConnected(end, routerIndexGlobal));
        text(hFinal.XData(nodeIdxInGfinal), hFinal.YData(nodeIdxInGfinal)+0.06, textInicial, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9, 'Color', 'k', 'FontWeight', 'bold');
        text(hFinal.XData(nodeIdxInGfinal), hFinal.YData(nodeIdxInGfinal)-0.06, textFinal, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 9, 'Color', 'k', 'FontWeight', 'bold');
    end
end
drawnow; saveas(fig3, fullfile(savePath, 'LOGR_TopologiaFinal.png')); close(fig3);

fig4 = figure('Name', 'Evolución de Estados (Color)', 'Position', [100 100 900 400], 'Visible', 'off');
normCount = zeros(simTimeTot, 1); alertCount = zeros(simTimeTot, 1); critCount = zeros(simTimeTot, 1);
for t_idx = 1:simTimeTot
    currentActiveRouters = sum(devicesConnected(t_idx, :) > 0 | ~isnan(latency(t_idx,:)));
    if t_idx <= simTimeCong; currentActiveRouters = numRoutersInit; end; if isnan(currentActiveRouters) || currentActiveRouters==0; currentActiveRouters=1; end
    activeStates = routerStatesColor(t_idx, 1:currentActiveRouters);
    normCount(t_idx) = sum(activeStates == "Normal");
    alertCount(t_idx) = sum(activeStates == "Alerta");
    critCount(t_idx) = sum(activeStates == "Critico");
end
plot(1:simTimeTot, movmean(normCount,3), 'g', 'LineWidth', 1.5); hold on;
plot(1:simTimeTot, movmean(alertCount,3), 'Color', [1 0.7 0], 'LineWidth', 1.5);
plot(1:simTimeTot, movmean(critCount,3), 'r', 'LineWidth', 1.5);
xline(60,'--k', 'Inicio optimización'); xlabel('Tiempo (s)'); ylabel('Número de Routers'); grid on;
legend('Normal (Verde)', 'Alerta (Naranja)', 'Critico (Rojo)');
title('Evolución de estados de color de los routers ACTIVOS');
drawnow; saveas(fig4, fullfile(savePath, 'LOGR_EvolucionEstadosColor.png')); close(fig4);

% --- INICIO DE MODIFICACIÓN: ANN -> Regresión Logística ---
fig5 = figure('Name', 'Métricas de eficiencia Reg. Logística', 'Position', [100 100 1000 400], 'Visible', 'off');
subplot(1,2,1);
metrics = [elapsedTime/1000, timeToOptimize, latencyReduction, lossReduction, effEnergy_pJ_per_bit, logrErrorPercent]; % Usar Error y pJ/bit
labels = {'T. Sim (s)', 'T. Opt (s)', 'Red. Lat %', 'Red. Perd %', 'Ef. Ene (pJ/bit)', 'Error Val. %'}; % Etiqueta de Error y pJ/bit
b5 = bar(metrics);
set(gca, 'XTickLabel', labels);
ylabel('Valores'); title('Métricas Reg. Logística (Enrutador)'); grid on;
ylim([0, max([metrics, 1])*1.2]);
text(b5.XEndPoints, b5.YData, sprintfc('%.2f', b5.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');

subplot(1,2,2);
theta = linspace(0, 2*pi, length(metrics)+1); rho = [metrics metrics(1)];
rho_norm = max(0, rho / max(rho) * 100); 
polarplot(theta, rho_norm, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b'); 
thetaticks(rad2deg(theta(1:end-1)));
thetaticklabels(labels);
rticks([25 50 75 100]); 
text(theta(1:end-1).*1.15, rho_norm(1:end-1).*1.15, sprintfc('%.1f', rho(1:end-1)),'HorizontalAlignment','center','FontSize',8);
title('Radar de métricas de Reg. Logística (Enrutador)');
drawnow; saveas(fig5, fullfile(savePath, 'LOGR_MetricasEficiencia.png')); close(fig5);
% --- FIN DE MODIFICACIÓN ---


fig6 = figure('Name', 'Evolución en Tiempo Real', 'Position', [100 100 1200 500], 'Visible', 'off');
mean_latency = zeros(simTimeTot, 1);
mean_throughput = zeros(simTimeTot, 1);
for t_idx = 1:simTimeTot
     if t_idx <= simTimeCong; currentActive = 1:numRoutersInit;
     else
         numActiveNow = numRoutersInit + floor((t_idx - simTimeCong -1)/addInterval)*routersPerInterval + min(routersPerInterval, mod(t_idx - simTimeCong -1, addInterval)+1);
         numActiveNow = min(numActiveNow, numRoutersMax);
         currentActive = 1:numActiveNow;
     end
     mean_latency(t_idx) = mean(latency(t_idx, currentActive),'omitnan');
     mean_throughput(t_idx) = mean(throughput(t_idx, currentActive),'omitnan');
end
subplot(1,2,1); plot(1:simTimeTot, mean_latency, 'r', 'LineWidth', 1.5); xline(60, '--k', 'Inicio optimización'); title('Evolución Latencia Promedio (Activos)'); xlabel('Tiempo (s)'); ylabel('ms'); grid on; xlim([0 simTimeTot]); ylim('auto');
subplot(1,2,2); plot(1:simTimeTot, mean_throughput, 'b', 'LineWidth', 1.5); xline(60, '--k', 'Inicio optimización'); title('Evolución Throughput Promedio (Activos)'); xlabel('Tiempo (s)'); ylabel('Gbps'); grid on; xlim([0 simTimeTot]); ylim('auto');
drawnow; saveas(fig6, fullfile(savePath, 'LOGR_EvolucionTiempoReal.png')); close(fig6);


% --- GUARDAR REPORTE PDF (en .txt) ---
reportPath = fullfile(savePath, 'Resultados_LOGR_Reporte.txt');
try
    fid = fopen(reportPath, 'w', 'n', 'UTF-8');
    if fid == -1; error('No se pudo crear el archivo de reporte en %s', reportPath); end
    disp(['Creando reporte de texto en: ', reportPath]);
    
    fprintf(fid, '==========================================================\n');
    fprintf(fid, ' REPORTE DE SIMULACIÓN - ENRUTADOR REGRESIÓN LOGÍSTICA\n');
    fprintf(fid, '==========================================================\n\n');
    fprintf(fid, 'Fecha de simulación: %s\n\n', datestr(now));

    fprintf(fid, '--- MÉTRICAS DE EFICIENCIA DE REG. LOGÍSTICA (Enrutador) ---\n');
    efficiencyOutput = evalc('disp(efficiencyMetrics)');
    fprintf(fid, '%s\n', efficiencyOutput);

    fprintf(fid, '--- RESULTADOS ANTES DE OPTIMIZACIÓN (Fase Congestión t=1-60) ---\n');
    beforeOutput = evalc('disp(resultsBefore)');
    fprintf(fid, '%s\n', beforeOutput);

    fprintf(fid, '--- RESULTADOS DESPUÉS DE OPTIMIZACIÓN (Fase Reg. Logística t=61-90) ---\n');
    afterOutput = evalc('disp(resultsAfter)');
    fprintf(fid, '%s\n', afterOutput);

    fprintf(fid, '--- %% DE MEJORA ENTRE CONGESTIÓN Y OPTIMIZACIÓN ---\n');
    comparisonOutput = evalc('disp(resultsComparison)');
    fprintf(fid, '%s\n', comparisonOutput);

    fprintf(fid, '--- DETALLES POR ROUTER (Estado Final t=90s) ---\n');
    detailsOutput = evalc('disp(routerDetails)');
    fprintf(fid, '%s\n', detailsOutput);
    
    fprintf(fid, '\n--- FIN DEL REPORTE ---');
    fclose(fid);
    
    disp('Reporte de texto guardado exitosamente.');
    disp('*** INSTRUCCIÓN: Abra el archivo .txt y use "Imprimir a PDF" para crear su reporte final. ***');
catch ME
    if fid ~= -1; fclose(fid); end
    disp('Error al guardar el reporte de texto:');
    disp(ME.message);
end
% --- FIN REPORTE ---

disp('--- SIMULACIÓN FINALIZADA ---');