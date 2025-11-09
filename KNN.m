% =========================================================================
% SIMULACIÓN DE RED 5G CONGESTIÓN Y OPTIMIZACIÓN (VERSIÓN KNN)
%
% BASADO EN EL SCRIPT FUNDAMENTAL (v5)
% 1. [MODELO] Enrutador ANN reemplazado por KNN ('fitcknn').
% 2. [MÉTRICA] Se usa 'kfoldLoss' para calcular Error (Fórmula 2)
%    y Eficacia (Fórmula 3) para una comparación justa.
% 3. [LÓGICA GRADUAL] Se mantiene la lógica de optimización gradual.
% 4. [rng(1)] Se mantiene para datos deterministas.
% 5. [RUTA] Guardado en carpeta KNN.
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


%% ===== KNN (Entrenamiento para ENRUTAMIENTO dinámico) ====
% --- INICIO DE MODIFICACIÓN: ANN -> KNN ---
disp('Preparando datos de entrenamiento para KNN (Enrutador)...');
datasetKNN = []; labelsKNN = []; % Labels para decisión de carga
critThresholdKNN = 46; % Umbrales para el OBJETIVO del enrutador
alertThresholdKNN = 26;
for d = 10:1:maxDevicesPerRouter
     for i = 1:10
        lat = 10 + (d^1.3)/10 + rand*3;
        loss = min(40, (d/3) + rand*3);
        thr = max(0.5, 9 - (d^1.1)/20 + rand*0.3);

        if d >= critThresholdKNN
            targetStateKNN = "Critico";
        elseif d >= alertThresholdKNN
            targetStateKNN = "Alerta";
        else
            targetStateKNN = "Normal";
        end
        % Features para KNN: [dispositivos, throughput, latencia, perdida]
        datasetKNN = [datasetKNN; d thr lat loss];
        labelsKNN = [labelsKNN; targetStateKNN];
    end
end
X_knn = datasetKNN; % Formato [Samples x Features] para fitcknn
Y_knn = categorical(labelsKNN);
classNamesKNN = categories(Y_knn); % Guardar nombres de clases para KNN

% Entrenamiento del clasificador KNN
disp('Entrenando modelo KNN (Enrutador)...');
% 'NumNeighbors'=5 es un valor común, 'Standardize'=true normaliza los datos
Mdl_KNN = fitcknn(X_knn, Y_knn, 'NumNeighbors', 5, 'Standardize', true, 'ClassNames', classNamesKNN);

% Validación cruzada del KNN para obtener Error (Fórmula 2)
disp('Calculando Error/Eficacia del modelo KNN...');
cvKNN = crossval(Mdl_KNN, 'KFold', 5);
knnErrorPercent = kfoldLoss(cvKNN) * 100; % Error (Fórmula 2)
knnAccuracyPercent = (1 - kfoldLoss(cvKNN)) * 100; % Eficacia (Fórmula 3)

disp(['Eficacia de validación cruzada KNN (Fórmula 3): ', num2str(knnAccuracyPercent, '%.2f'), '%']);
disp(['Error de validación cruzada KNN (Fórmula 2): ', num2str(knnErrorPercent, '%.2f'), '%']);
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

%% ===== CONFIGURACIÓN DE GUARDADO (KNN) =====
% --- RUTA ACTUALIZADA ---
savePath = 'C:\Users\jordy\Documents\Resultados\KNN\';
% --- FIN ACTUALIZACIÓN ---
if ~exist(savePath, 'dir')
    mkdir(savePath);
end
disp('--- INICIANDO SIMULACIÓN KNN ---');

%% ===== FIGURA PRINCIPAL =====
fig1 = figure('Name', 'Simulación Red 5G - Congestión y Optimización (KNN)', 'Position', [50 50 1600 800]);

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

            % --- INICIO DE MODIFICACIÓN: ANN -> KNN ---
            if isNewRouter && devicesConnected(prevStateTime, r) == 0
                 features_knn = [0, 10, 5, 0]; % Estado ideal [1 x 4]
                 predKNN_Enrutador = "Normal"; 
            else
                 prevDev = devicesConnected(prevStateTime,r);
                 prevThr = fillmissing(throughput(prevStateTime,r),'constant', 10); 
                 prevLat = fillmissing(latency(prevStateTime,r),'constant', 5); 
                 prevLoss = fillmissing(lostPackets(prevStateTime,r),'constant', 0); 
                 features_knn = [prevDev, prevThr, prevLat, prevLoss]; % Formato [1 x 4]

                 % Predecir el estado con KNN (Enrutador)
                 predKNN_Enrutador = predict(Mdl_KNN, features_knn);
            end
            % --- FIN DE MODIFICACIÓN ---

            % Política de enrutamiento GRADUAL (NUEVO ESTÁNDAR)
            if strcmp(predKNN_Enrutador, 'Critico')
                numTotal = randi([10, 20]); 
            elseif strcmp(predKNN_Enrutador, 'Alerta')
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
    % --- INICIO DE MODIFICACIÓN: ANN -> KNN ---
    if t <= simTimeCong; title(sprintf('Congestión - Segundo %d / %d (%d Routers)', t, simTimeTot, routersActivos));
    else; title(sprintf('Optimización (KNN) - Segundo %d / %d (%d Routers)', t, simTimeTot, routersActivos)); end
    % --- FIN DE MODIFICACIÓN ---

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
disp('--- RESULTADOS DESPUÉS DE OPTIMIZACIÓN (Fase KNN t=61-90) ---'); disp(resultsAfter);
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


%% ===== MÉTRICAS DE EFICIENCIA DEL KNN (Fórmulas 2 y 3) =====
% --- INICIO DE MODIFICACIÓN: ANN -> KNN ---
timeToOptimize = simTimeOpt;
latencyReduction = improvement(1);
lossReduction = improvement(3);

avgEnergyJoule = avgEnergyAfter / 1000; % (mJ/s) -> (J/s)
avgBitsPerSec = avgThrAfter * 1e9;    % (Gbit/s) -> (bit/s)
if avgThrAfter > 0 && avgEnergyJoule > 0; effEnergy_J_per_bit = avgEnergyJoule / avgBitsPerSec; else; effEnergy_J_per_bit = 0; end
effEnergy_pJ_per_bit = effEnergy_J_per_bit * 1e12; % Convertir J/bit a pJ/bit

knnAccuracyPercent = knnAccuracyPercent; % Tu Fórmula 3 (Eficacia)
knnErrorPercent = knnErrorPercent; % Tu Fórmula 2 (Error)

efficiencyMetrics = table(elapsedTime, timeToOptimize, latencyReduction, lossReduction, effEnergy_pJ_per_bit, knnErrorPercent, knnAccuracyPercent, ...
    'VariableNames', {'TiempoEnrutamiento_ms', 'TiempoOptimiz_s', 'ReduccionLatencia_p', 'ReduccionPerdida_p', 'EficienciaEnergetica_pJbit', 'ErrorValidacion_KNN_p', 'EficaciaValidacion_KNN_p'});
disp('--- MÉTRICAS DE EFICIENCIA DEL KNN (Enrutador) ---'); disp(efficiencyMetrics);
% --- FIN DE MODIFICACIÓN ---


%% ===== GUARDAR RESULTADOS DEL KNN =====
% --- INICIO DE MODIFICACIÓN: ANN -> KNN ---
Resultados_KNN.Tablas.Before = resultsBefore;
Resultados_KNN.Tablas.After = resultsAfter;
Resultados_KNN.Tablas.Comparison = resultsComparison;
Resultados_KNN.Tablas.RouterDetails = routerDetails;
Resultados_KNN.Tablas.Efficiency = efficiencyMetrics;
Resultados_KNN.Series.Latency = latency;
Resultados_KNN.Series.Throughput = throughput;
Resultados_KNN.Series.Loss = lostPackets;
Resultados_KNN.Series.Energy = energyConsumption;
Resultados_KNN.Series.StatesColor = routerStatesColor;
Resultados_KNN.Modelo.KNN_Model = Mdl_KNN;
Resultados_KNN.Modelo.CV_Info = cvKNN;

save(fullfile(savePath, 'Resultados_KNN.mat'), 'Resultados_KNN');
writetable(resultsBefore, fullfile(savePath, 'Resultados_KNN_Before.csv'));
writetable(resultsAfter, fullfile(savePath, 'Resultados_KNN_After.csv'));
writetable(resultsComparison, fullfile(savePath, 'Resultados_KNN_Comparison.csv'));
writetable(routerDetails, fullfile(savePath, 'Resultados_KNN_RouterDetails.csv'));
writetable(efficiencyMetrics, fullfile(savePath, 'Resultados_KNN_Efficiency.csv'));

disp(['--- TODOS LOS RESULTADOS DEL KNN SE HAN GUARDADO EN: ', savePath, ' ---']);
% --- FIN DE MODIFICACIÓN ---

%% ===== GRÁFICOS COMPARATIVOS (FIGURAS 2-5) =====

% La figura 1 (en vivo) se deja abierta
fig2 = figure('Name', 'Comparación Antes vs Después', 'Position', [100 100 1200 600], 'Visible', 'off');
subplot(2,2,1); b1 = bar([avgLatencyBefore avgLatencyAfter], 'r'); ylabel('ms'); title('Latencia promedio (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgLatencyBefore, avgLatencyAfter, 1])*1.2]); text(b1.XEndPoints, b1.YEndPoints, sprintfc('%.2f ms', b1.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
subplot(2,2,2); b2 = bar([avgThrBefore avgThrAfter], 'b'); ylabel('Gbps'); title('Throughput promedio (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgThrBefore, avgThrAfter, 1])*1.2]); text(b2.XEndPoints, b2.YEndPoints, sprintfc('%.2f Gbps', b2.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
subplot(2,2,3); b3 = bar([avgLossBefore avgLossAfter], 'FaceColor', [1 0.5 0]); ylabel('%'); title('Pérdida de paquetes (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgLossBefore, avgLossAfter, 1])*1.2]); text(b3.XEndPoints, b3.YEndPoints, sprintfc('%.2f %%', b3.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
subplot(2,2,4); b4 = bar([avgEnergyBefore avgEnergyAfter], 'g'); ylabel('mJ'); title('Consumo energético (General)'); set(gca, 'XTickLabel', {'Congestión', 'Optimización'}); ylim([0, max([avgEnergyBefore, avgEnergyAfter, 0.1])*1.2]); text(b4.XEndPoints, b4.YEndPoints, sprintfc('%.2f mJ', b4.YData), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
drawnow; saveas(fig2, fullfile(savePath, 'KNN_ComparacionAntesDespues.png')); close(fig2);


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
drawnow; saveas(fig3, fullfile(savePath, 'KNN_TopologiaFinal.png')); close(fig3);

fig4 = figure('Name', 'Evolución de Estados (Color)', 'Position', [100 100 900 400], 'Visible', 'off');
normCount = zeros(simTimeTot, 1); alertCount = zeros(simTimeTot, 1); critCount = zeros(simTimeTot, 1);
for t_idx = 1:simTimeTot
    currentActiveRouters = sum(devicesConnected(t_idx, :) > 0 | ~isnan(latency(t_idx,:)));
    if t_idx <= simTimeCong; currentActiveRouters = numRoutersInit; end; if isnan(currentActiveRouters) || currentActiveRouters==0; currentActiveRouters=1; end
    activeStates = routerStatesColor(t_idx, 1:currentActiveRouters);
    normCount(t_idx) = sum(activeStates == "Normal"); alertCount(t_idx) = sum(activeStates == "Alerta"); critCount(t_idx) = sum(activeStates == "Critico");
end
plot(1:simTimeTot, movmean(normCount,3), 'g', 'LineWidth', 1.5); hold on;
plot(1:simTimeTot, movmean(alertCount,3), 'Color', [1 0.7 0], 'LineWidth', 1.5);
plot(1:simTimeTot, movmean(critCount,3), 'r', 'LineWidth', 1.5);
xline(60,'--k', 'Inicio optimización'); xlabel('Tiempo (s)'); ylabel('Número de Routers'); grid on;
legend('Normal (Verde)', 'Alerta (Naranja)', 'Critico (Rojo)');
title('Evolución de estados de color de los routers ACTIVOS');
drawnow; saveas(fig4, fullfile(savePath, 'KNN_EvolucionEstadosColor.png')); close(fig4);

% --- INICIO DE MODIFICACIÓN: ANN -> KNN ---
fig5 = figure('Name', 'Métricas de eficiencia KNN', 'Position', [100 100 1000 400], 'Visible', 'off');
subplot(1,2,1);
metrics = [elapsedTime/1000, timeToOptimize, latencyReduction, lossReduction, effEnergy_pJ_per_bit, knnErrorPercent]; % Usar Error y pJ/bit
labels = {'T. Sim (s)', 'T. Opt (s)', 'Red. Lat %', 'Red. Perd %', 'Ef. Ene (pJ/bit)', 'Error Val. %'}; % Etiqueta de Error y pJ/bit
b5 = bar(metrics);
set(gca, 'XTickLabel', labels);
ylabel('Valores'); title('Métricas KNN (Enrutador)'); grid on;
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
title('Radar de métricas del KNN (Enrutador)');
drawnow; saveas(fig5, fullfile(savePath, 'KNN_MetricasEficiencia.png')); close(fig5);
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
drawnow; saveas(fig6, fullfile(savePath, 'KNN_EvolucionTiempoReal.png')); close(fig6);


% --- GUARDAR REPORTE PDF (en .txt) ---
reportPath = fullfile(savePath, 'Resultados_KNN_Reporte.txt');
try
    fid = fopen(reportPath, 'w', 'n', 'UTF-8');
    if fid == -1; error('No se pudo crear el archivo de reporte en %s', reportPath); end
    disp(['Creando reporte de texto en: ', reportPath]);

    % --- Escribir todas las tablas de resultados al archivo ---
    fprintf(fid, '==========================================================\n');
    fprintf(fid, ' REPORTE DE SIMULACIÓN - ENRUTADOR KNN\n');
    fprintf(fid, '==========================================================\n\n');
    fprintf(fid, 'Fecha de simulación: %s\n\n', datestr(now));

    fprintf(fid, '--- MÉTRICAS DE EFICIENCIA DEL KNN (Enrutador) ---\n');
    efficiencyOutput = evalc('disp(efficiencyMetrics)');
    fprintf(fid, '%s\n', efficiencyOutput);

    fprintf(fid, '--- RESULTADOS ANTES DE OPTIMIZACIÓN (Fase Congestión t=1-60) ---\n');
    beforeOutput = evalc('disp(resultsBefore)');
    fprintf(fid, '%s\n', beforeOutput);

    fprintf(fid, '--- RESULTADOS DESPUÉS DE OPTIMIZACIÓN (Fase KNN t=61-90) ---\n');
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