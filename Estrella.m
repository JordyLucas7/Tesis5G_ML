% =========================================================================
% SIMULACIÓN DE RED 5G - 15 ROUTERS ESTRELLA (CORREGIDO Y ESTANDARIZADO v7)
%
% OBJETIVO: Simular la FASE DE CONGESTIÓN (t=1-60) con topología de Estrella.
%
% CAMBIOS REALIZADOS (v7 - Lógica de Topología Real):
% 1. [LÓGICA DE CUELLO DE BOTELLA] Se añade lógica donde la latencia y
%    pérdida del nodo central (Router_1) se SUMA a la de todos los
%    demás nodos, simulando un punto de fallo central.
% 2. [SIN rng(1)] Se elimina 'rng(1)' para permitir variabilidad natural.
% 3. [Fórmulas] Se usan las fórmulas de métricas unificadas (v5).
% 4. [Reporte] Se usa el guardado de gráficos y reporte .txt estándar.
% =========================================================================

clear; clc; close all;
% --- 'rng(1)' SE ELIMINA INTENCIONALMENTE para que esta simulación
% --- sea diferente a la de Malla y Árbol.

%% ===== PARAMETROS =====
numRouters = 15;
maxDevicesPerRouter = 70; % Estandarizado a 70
simTime = 60; % Solo Fase de Congestión

% Carpeta destino para guardar resultados
outputDir = 'C:\Users\jordy\Documents\Resultados\Topologías\TESTRELLA\';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Crear routers
routers = strcat("Router_", string(1:numRouters));

% Crear grafo base solo con routers
G = graph();
G = addnode(G, routers);

% ================== CONECTAR TOPOLOGÍA ESTRELLA ==================
% Router_1 es el nodo central
nodoCentral = routers(1);
for j = 2:numRouters
    G = addedge(G, nodoCentral, routers(j));
end

%% ===== CLASIFICADOR DE COLOR (Árbol de Decisión Simple) ====
% (Estandarizado con los scripts de ML)
trainDataColor = []; labelsColor = [];
critThresholdColor = 46; % >= 46 Crítico (Rojo)
alertThresholdColor = 26; % 26-45 Alerta (Naranja)
% <= 25 Normal (Verde)
for d = 1:maxDevicesPerRouter
    if d >= critThresholdColor; stateColor = "Critico";
    elseif d >= alertThresholdColor; stateColor = "Alerta";
    else; stateColor = "Normal"; end
    trainDataColor = [trainDataColor; d];
    labelsColor = [labelsColor; stateColor];
end
labelsColor = categorical(labelsColor);
Mdl_Color = fitctree(trainDataColor, labelsColor);


%% ===== INICIALIZAR VARIABLES =====
latency = NaN(simTime, numRouters);
throughput = NaN(simTime, numRouters);
lostPackets = NaN(simTime, numRouters);
energyConsumption = NaN(simTime, numRouters);
packetsSent = zeros(simTime, numRouters);
packetsReceived = zeros(simTime, numRouters);
devicesConnected = zeros(simTime, numRouters);
routerStatesColor = strings(simTime, numRouters);
finalLaptops = zeros(numRouters,1);
finalPhones = zeros(numRouters,1);
laptopColor = [0 0.6 0]; phoneColor  = [1 0 0];

%% ================== SIMULACIÓN ==================
fig1 = figure('Name','Simulación Red 5G - Estrella (Lógica Corregida)','Position',[100 100 1200 700]);
tic;

for t = 1:simTime
    Gtemp = G;
    
    % --- Bucle 1: Calcular métricas base para cada router ---
    for r = 1:numRouters
        % --- Carga Estandarizada ---
        numTotal = round((t/simTime) * maxDevicesPerRouter);
        if numTotal == 0; numTotal = 1; end % Evitar 0
        devicesConnected(t,r) = numTotal;
        numLaptops = randi([round(0.4*numTotal), round(0.6*numTotal)]); % Split 40/60
        numPhones  = numTotal - numLaptops;

        if numLaptops > 0
            laptops = strcat("L_", string(r), "_", string(1:numLaptops));
            Gtemp = addnode(Gtemp, laptops);
            for L = 1:numLaptops; Gtemp = addedge(Gtemp, routers(r), laptops(L)); end
        end
        if numPhones > 0
            phones  = strcat("P_", string(r), "_", string(1:numPhones));
            Gtemp = addnode(Gtemp, phones);
            for P = 1:numPhones; Gtemp = addedge(Gtemp, routers(r), phones(P)); end
        end

        if t == simTime; finalLaptops(r) = numLaptops; finalPhones(r) = numPhones; end

        % --- Tráfico Estandarizado ---
        if numTotal > 0
            trafficLaptops = poissrnd(60, [1 numLaptops]);
            trafficPhones = poissrnd(90, [1 numPhones]) + randn(1,numPhones)*15;
            trafficPhones(trafficPhones < 0) = 0;
            currentTraffic = sum(trafficLaptops) + sum(trafficPhones);
            if isempty(currentTraffic); currentTraffic = 0; end
            packetsSent(t,r) = currentTraffic / 10;
        else
            packetsSent(t,r) = 0;
        end

        % --- Fórmulas de Métricas Estandarizadas (Base) ---
        if numTotal > 0
            packetsReceived(t,r) = packetsSent(t,r) * (0.8 - 0.3*(numTotal/maxDevicesPerRouter));
            latency(t,r) = max(5, 7 + (numTotal^1.2)/12 + rand*1.5);
            lostPackets(t,r) = max(0, min(25, (numTotal/3.5) + rand*1.5));
            throughput(t,r) = max(1, 9 - (numTotal^1.0)/25 + rand*0.2);
            energyConsumption(t,r) = max(0.4, 0.7 + 0.01*numTotal + rand*0.1);
        else
            packetsReceived(t,r) = 0;
            latency(t,r) = NaN; lostPackets(t,r) = NaN; throughput(t,r) = NaN; energyConsumption(t,r) = NaN;
        end
    end % Fin bucle for r

    % --- INICIO DE LÓGICA DE TOPOLOGÍA (CUELLO DE BOTELLA) ---
    % La latencia y pérdida del nodo central (Router 1) impacta a todos.
    latencia_central = latency(t,1);
    perdida_central = lostPackets(t,1);
    
    for r = 2:numRouters % Iterar solo sobre los routers "hoja"
        latency(t,r) = latency(t,r) + latencia_central;
        lostPackets(t,r) = lostPackets(t,r) + perdida_central;
    end
    % Asegurarse de que la pérdida no supere un máximo (ej. 100%)
    lostPackets(t,:) = min(lostPackets(t,:), 100);
    % --- FIN DE LÓGICA DE TOPOLOGÍA ---

    % --- Clasificación de Color (se basa en la carga, no en la latencia acumulada) ---
    for r = 1:numRouters
        predColor = predict(Mdl_Color, devicesConnected(t,r));
        routerStatesColor(t,r) = string(predColor);
    end

    % ===== VISUALIZACIÓN =====
    subplot(2,2,[1 3]); cla;
    h = plot(Gtemp,'Layout','force','NodeLabel',{}); % Layout de Estrella
    
    nodeIndicesPC = findnode(Gtemp, Gtemp.Nodes.Name(contains(Gtemp.Nodes.Name,"L_")));
    if ~isempty(nodeIndicesPC); highlight(h, nodeIndicesPC, 'NodeColor', laptopColor, 'MarkerSize', 3); end
    nodeIndicesMovil = findnode(Gtemp, Gtemp.Nodes.Name(contains(Gtemp.Nodes.Name,"P_")));
    if ~isempty(nodeIndicesMovil); highlight(h, nodeIndicesMovil, 'NodeColor', phoneColor, 'MarkerSize', 3); end

    for r = 1:numRouters
        state = routerStatesColor(t,r); nodeIdx = findnode(Gtemp, routers(r));
        if nodeIdx > 0; if state == "Normal"; color = [0 0.7 0]; elseif state == "Alerta"; color = [1 0.7 0]; else; color = [0.85 0.1 0.1]; end
            highlight(h, nodeIdx, 'NodeColor', color, 'MarkerSize', 7); end
    end
    title(sprintf('Topología en Estrella - Segundo %d',t));

    subplot(2,2,2); cla; plot(1:t,mean(latency(1:t,:),2,'omitnan'),'b','LineWidth',1.5); title('Latencia promedio de la red (Acumulativa)'); xlabel('Tiempo (s)'); ylabel('ms'); grid on; xlim([0 simTime]);
    subplot(2,2,4); cla; plot(1:t,mean(lostPackets(1:t,:),2,'omitnan'),'r','LineWidth',1.5); title('Pérdida de paquetes promedio (%) (Acumulativa)'); xlabel('Tiempo (s)'); ylabel('%'); grid on; xlim([0 simTime]);

    drawnow; pause(0.05);
end
elapsedTime = toc*1000;
disp(['Tiempo total de simulación: ', num2str(elapsedTime/1000, '%.2f'), ' segundos']);

%% ================== RESULTADOS (Estandarizados) ==================
avgLatency = mean(mean(latency,'omitnan'));
avgThroughput = mean(mean(throughput,'omitnan'));
avgLost = mean(mean(lostPackets,'omitnan'));
avgEnergy = mean(mean(energyConsumption,'omitnan'));

resultsGeneral = table(avgLatency, avgThroughput, avgLost, avgEnergy, ...
    'VariableNames', {'Latencia_Promedio_ms', 'Throughput_Promedio_Gbps', 'Perdida_Promedio_p', 'Energia_Promedio_mJ'});

routerNames = routers';
finalRouterState = table(routerNames, finalLaptops, finalPhones, ...
    devicesConnected(end,:)', latency(end,:)', throughput(end,:)', lostPackets(end,:)', energyConsumption(end,:)', ...
    'VariableNames', {'Router','Laptops','Telefonos','Dispositivos_Finales','Latencia_ms_Final','Throughput_Gbps_Final','Perdida_p_Final','Energia_mJ_Final'});

disp('--- RESULTADOS GENERALES DE LA TOPOLOGÍA (Promedio t=1-60) ---');
disp(resultsGeneral);
disp('--- ESTADO FINAL DE ROUTERS (t=60s) ---');
disp(finalRouterState);

%% ===== GUARDAR REPORTE PDF (en .txt) =====
reportPath = fullfile(outputDir, 'Reporte_Estrella.txt');
try
    fid = fopen(reportPath, 'w', 'n', 'UTF-8');
    if fid == -1; error('No se pudo crear el archivo de reporte en %s', reportPath); end
    disp(['Creando reporte de texto en: ', reportPath]);
    
    fprintf(fid, '==========================================================\n');
    fprintf(fid, ' REPORTE DE SIMULACIÓN - TOPOLOGÍA ESTRELLA (CORREGIDO)\n');
    fprintf(fid, '==========================================================\n\n');
    fprintf(fid, 'Fecha de simulación: %s\n\n', datestr(now));
    fprintf(fid, '--- RESULTADOS GENERALES DE LA TOPOLOGÍA (Promedio t=1-60) ---\n');
    reportOutput = evalc('disp(resultsGeneral)');
    fprintf(fid, '%s\n', reportOutput);
    fprintf(fid, '--- ESTADO FINAL DE ROUTERS (t=60s) ---\n');
    reportOutput = evalc('disp(finalRouterState)');
    fprintf(fid, '%s\n', reportOutput);
    fprintf(fid, '\n--- FIN DEL REPORTE ---');
    fclose(fid);
    disp('Reporte de texto guardado exitosamente.');
catch ME
    if fid ~= -1; fclose(fid); end
    disp('Error al guardar el reporte de texto:');
    disp(ME.message);
end

%% ================== GRÁFICOS (Estandarizados) ==================
try; close(fig1); catch; end
fig2 = figure('Name', 'Resultados Promedio - Estrella', 'Position', [100 100 1200 600], 'Visible', 'off');
subplot(2,2,1); bar(avgLatency,'r'); title('Latencia Promedio (ms)'); ylabel('ms');
subplot(2,2,2); bar(avgThroughput,'b'); title('Throughput Promedio (Gbps)'); ylabel('Gbps');
subplot(2,2,3); bar(avgLost,'FaceColor',[1 0.5 0]); title('Pérdida Paquetes Promedio (%)'); ylabel('%');
subplot(2,2,4); bar(avgEnergy,'g'); title('Consumo Energético Promedio (mJ)'); ylabel('mJ');
drawnow; saveas(fig2, fullfile(outputDir, 'Estrella_Resultados_Promedio.png')); close(fig2);

fig3 = figure('Name', 'Topologia Final - Estrella', 'Position', [100 100 900 600], 'Visible', 'off');
hFinal = plot(G,'Layout','force','NodeLabel',{});
title('Topologia Final - Estado en t=60s');
for r = 1:numRouters
    routerName = routers(r); finalState = routerStatesColor(end, r);
    if finalState == "Normal"; color = [0 0.7 0]; elseif finalState == "Alerta"; color = [1 0.7 0]; else; color = [0.85 0.1 0.1]; end
    nodeIdxInG = findnode(G, routerName);
    if nodeIdxInG > 0
        highlight(hFinal, nodeIdxInG, 'NodeColor', color, 'MarkerSize', 8);
        textFinal = sprintf('%s\n%s (%dd)', routerName, finalState, devicesConnected(end, r));
        text(hFinal.XData(nodeIdxInG), hFinal.YData(nodeIdxInG)-0.05, textFinal, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 9, 'Color', 'k', 'FontWeight', 'bold');
    end
end
drawnow; saveas(fig3, fullfile(outputDir, 'Estrella_TopologiaFinal.png')); close(fig3);

fig4 = figure('Name', 'Evolución de Estados (Color) - Estrella', 'Position', [100 100 900 400], 'Visible', 'off');
normCount = sum(routerStatesColor=="Normal", 2); alertCount = sum(routerStatesColor=="Alerta", 2); critCount = sum(routerStatesColor=="Critico", 2);
plot(1:simTime, movmean(normCount,3), 'g', 'LineWidth', 1.5); hold on;
plot(1:simTime, movmean(alertCount,3), 'Color', [1 0.7 0], 'LineWidth', 1.5);
plot(1:simTime, movmean(critCount,3), 'r', 'LineWidth', 1.5);
xlabel('Tiempo (s)'); ylabel('Número de Routers'); grid on;
legend('Normal (Verde)', 'Alerta (Naranja)', 'Critico (Rojo)');
title('Evolución de estados de color de los routers');
drawnow; saveas(fig4, fullfile(outputDir, 'Estrella_EvolucionEstadosColor.png')); close(fig4);

fig6 = figure('Name', 'Evolución en Tiempo Real - Estrella', 'Position', [100 100 1200 500], 'Visible', 'off');
mean_latency = mean(latency, 2,'omitnan'); mean_throughput = mean(throughput, 2,'omitnan');
subplot(1,2,1); plot(1:simTime, mean_latency, 'r', 'LineWidth', 1.5); title('Evolución Latencia Promedio (Activos)'); xlabel('Tiempo (s)'); ylabel('ms'); grid on; xlim([0 simTime]); ylim('auto');
subplot(1,2,2); plot(1:simTime, mean_throughput, 'b', 'LineWidth', 1.5); title('Evolución Throughput Promedio (Activos)'); xlabel('Tiempo (s)'); ylabel('Gbps'); grid on; xlim([0 simTime]); ylim('auto');
drawnow; saveas(fig6, fullfile(outputDir, 'Estrella_EvolucionTiempoReal.png')); close(fig6);

disp('--- SIMULACIÓN DE TOPOLOGÍA ESTRELLA CORREGIDA Y FINALIZADA ---');