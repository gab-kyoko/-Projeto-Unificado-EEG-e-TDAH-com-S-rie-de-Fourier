%% ANÁLISE ESPECTRAL E WAVELET DE SINAIS EEG PARA DETECÇÃO DE TDAH
% Autor: Gemini AI (com base nas solicitações do usuário)
% Data: 16/06/2025
% Descrição: Este script realiza uma análise completa de dados de EEG,
% comparando grupos de TDAH e Controle. Inclui pré-processamento,
% análise de frequência (FFT), tempo-frequência (Wavelet - CWT) e
% visualização de espectrogramas.

function analise_eeg_completa()
    % Limpa o ambiente e fecha figuras
    clear; clc; close all;

    % --- 1. CONFIGURAÇÕES GERAIS ---
    params = struct();
    params.fs = 256; % Frequência de amostragem padrão (Hz)
    params.files = { % Arquivos do dataset original
        'FADHD.mat', 'FC.mat', 'MADHD.mat', 'MC.mat'
    };
    params.group_labels = {'F-ADHD', 'F-Ctrl', 'M-ADHD', 'M-Ctrl'};
    params.group_types = {'ADHD', 'Ctrl', 'ADHD', 'Ctrl'};
    params.gender_types = {'F', 'F', 'M', 'M'};

    % Bandas de frequência
    bands = struct();
    bands.delta = [0.5, 4];
    bands.theta = [4, 8];
    bands.alpha = [8, 13];
    bands.beta = [13, 30];
    bands.gamma = [30, 50];

    fprintf('=== ANÁLISE AVANÇADA DE SINAIS DE EEG (TDAH) ===\n');

    % --- 2. EXECUÇÃO DA ANÁLISE ---
    try
        % Carregar os datasets
        [all_data, fs] = load_datasets(params.files, params.fs);
        params.fs = fs; % Atualiza a frequência de amostragem real

        % Análise multi-grupo
        fprintf('\n--- INICIANDO ANÁLISE (FFT & CWT) ---\n');
        results = analyze_all_groups(all_data, fs, bands, params.group_labels, params.group_types);

        % Gerar visualizações
        fprintf('\n--- GERANDO VISUALIZAÇÕES ---\n');
        create_visualizations(results);
        
        % --- NOVO: Gerar espectrogramas de exemplo ---
        fprintf('\n--- GERANDO ESPECTROGRAMAS ---\n');
        create_spectrogram_visualization(all_data, fs);
        % --- FIM NOVO ---

        % Imprimir relatório estatístico
        fprintf('\n--- RELATÓRIO ESTATÍSTICO ---\n');
        print_report(results);

        fprintf('\n=== ANÁLISE CONCLUÍDA COM SUCESSO ===\n');

    catch ME
        fprintf('\n!!! OCORREU UM ERRO DURANTE A EXECUÇÃO !!!\n');
        fprintf('Erro: %s\n', ME.message);
        fprintf('Arquivo: %s, Linha: %d\n', ME.stack(1).file, ME.stack(1).line);
        rethrow(ME);
    end
end

%% --- FUNÇÕES DE CARREGAMENTO DE DADOS ---
function [all_data, fs_estimated] = load_datasets(files, expected_fs)
    all_data = cell(length(files), 1);
    fs_values = [];
    for i = 1:length(files)
        filename = files{i};
        if ~exist(filename, 'file')
            warning('Arquivo %s não encontrado. Pulando...', filename);
            all_data{i} = {};
            continue;
        end
        fprintf('\nCarregando dados de: %s\n', filename);
        data_struct = load(filename);
        field_names = fieldnames(data_struct);
        dataset = data_struct.(field_names{1}); % Assume o primeiro campo
        
        [group_data, fs_group] = process_group_data(dataset, expected_fs);
        all_data{i} = group_data;
        if ~isempty(fs_group), fs_values = [fs_values, fs_group]; end
    end
    fs_estimated = round(median(fs_values));
    fprintf('\nFrequência de amostragem final: %d Hz\n', fs_estimated);
end

function [group_data, fs_est] = process_group_data(dataset, fs)
    group_data = {};
    fs_values = [];
    if iscell(dataset)
        for i = 1:length(dataset)
            subject_data = dataset{i};
            if size(subject_data, 1) > size(subject_data, 2)
                subject_data = subject_data'; % Garante formato (canais x amostras)
            end
            group_data{end+1} = subject_data;
            fs_values = [fs_values, size(subject_data, 2) / 30]; % Estima fs (30s de gravação)
        end
    end
    fs_est = median(fs_values);
end

%% --- FUNÇÕES DE ANÁLISE DE SINAL ---
function results = analyze_all_groups(all_data, fs, bands, group_labels, group_types)
    results = struct('fft', [], 'cwt', []);
    tb_ratios_fft = cell(length(all_data), 1);
    tb_ratios_cwt = cell(length(all_data), 1);

    for g = 1:length(all_data)
        group_data = all_data{g};
        fprintf('Analisando Grupo: %s (%d sujeitos)\n', group_labels{g}, length(group_data));
        
        for s = 1:length(group_data)
            subject_data = group_data{s};
            n_channels = size(subject_data, 1);
            for ch = 1:min(n_channels, 2) % Limita a 2 canais
                signal = subject_data(ch, :);
                
                % Análise FFT
                [~, tb_fft] = analyze_spectrum(signal, fs, bands);
                if ~isnan(tb_fft), tb_ratios_fft{g} = [tb_ratios_fft{g}; tb_fft]; end
                
                % Análise Wavelet (CWT)
                [~, tb_cwt] = analyze_wavelet(signal, fs, bands);
                if ~isnan(tb_cwt), tb_ratios_cwt{g} = [tb_ratios_cwt{g}; tb_cwt]; end
            end
        end
    end
    
    % Armazena resultados e calcula estatísticas
    results.fft.tb_ratios_by_group = tb_ratios_fft;
    results.cwt.tb_ratios_by_group = tb_ratios_cwt;
    results.fft.stats = calculate_statistics(tb_ratios_fft, group_types);
    results.cwt.stats = calculate_statistics(tb_ratios_cwt, group_types);
    results.group_labels = group_labels;
end

function [power_bands, theta_beta_ratio] = analyze_spectrum(signal, fs, bands)
    % Filtro passa-banda para limpeza do sinal
    bpFilt = designfilt('bandpassiir','FilterOrder',4, 'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',50, 'SampleRate',fs);
    signal_filtered = filter(bpFilt, signal);
    signal_detrended = detrend(signal_filtered);
    
    % FFT com janela de Hanning
    N = length(signal_detrended);
    window = hanning(N)';
    Y = fft(signal_detrended .* window);
    P2 = abs(Y/N);
    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    psd = P1.^2;
    freqs = fs*(0:floor(N/2))/N;
    
    % Calcula potência nas bandas
    power_bands = calculate_band_power(psd, freqs, bands);
    if power_bands.beta > 0
        theta_beta_ratio = power_bands.theta / power_bands.beta;
    else
        theta_beta_ratio = NaN;
    end
end

function [power_bands, theta_beta_ratio] = analyze_wavelet(signal, fs, bands)
    % --- CORRIGIDO ---
    % Força a CWT a analisar no intervalo de frequência de interesse (0.5Hz a 50Hz)
    [cfs, freqs] = cwt(signal, fs, 'FrequencyLimits', [0.5 50]);    %Plotar no dominio do tempo
    
    psd = abs(cfs).^2; % Potência instantânea
    mean_psd = mean(psd, 2); % Média da potência no tempo
    
    % Calcula potência nas bandas
    power_bands = calculate_band_power(mean_psd, freqs, bands);
    if power_bands.beta > 0
        theta_beta_ratio = power_bands.theta / power_bands.beta;
    else
        theta_beta_ratio = NaN;
    end
end

function power_bands = calculate_band_power(psd, freqs, bands)
    power_bands = struct();
    band_names = fieldnames(bands);
    for i = 1:length(band_names)
        band_name = band_names{i};
        band_range = bands.(band_name);
        idx_band = freqs >= band_range(1) & freqs <= band_range(2);
        if any(idx_band)
            power_bands.(band_name) = trapz(freqs(idx_band), psd(idx_band));
        else
            power_bands.(band_name) = 0;
        end
    end
end

%% --- FUNÇÕES DE ESTATÍSTICA E RELATÓRIO ---
function stats = calculate_statistics(tb_ratios_by_group, group_types)
    stats = struct();
    adhd_data = [];
    ctrl_data = [];
    for g = 1:length(group_types)
        if ~isempty(tb_ratios_by_group{g})
            if strcmp(group_types{g}, 'ADHD')
                adhd_data = [adhd_data; tb_ratios_by_group{g}];
            elseif strcmp(group_types{g}, 'Ctrl')
                ctrl_data = [ctrl_data; tb_ratios_by_group{g}];
            end
        end
    end
    
    if ~isempty(adhd_data) && ~isempty(ctrl_data)
        [~, p] = ttest2(adhd_data, ctrl_data, 'Vartype', 'unequal');
        stats.adhd_mean = mean(adhd_data);
        stats.ctrl_mean = mean(ctrl_data);
        stats.p_value = p;
    else
        stats.adhd_mean = NaN;
        stats.ctrl_mean = NaN;
        stats.p_value = NaN;
    end
end

function print_report(results)
    fprintf('Análise FFT:\n');
    fprintf('  Razão T/B Média (TDAH): %.3f\n', results.fft.stats.adhd_mean);
    fprintf('  Razão T/B Média (Controle): %.3f\n', results.fft.stats.ctrl_mean);
    fprintf('  Teste-t (p-valor): %.4f\n', results.fft.stats.p_value);
    
    fprintf('Análise Wavelet (CWT):\n');
    fprintf('  Razão T/B Média (TDAH): %.3f\n', results.cwt.stats.adhd_mean);
    fprintf('  Razão T/B Média (Controle): %.3f\n', results.cwt.stats.ctrl_mean);
    fprintf('  Teste-t (p-valor): %.4f\n', results.cwt.stats.p_value);
end

%% --- FUNÇÕES DE VISUALIZAÇÃO ---
function create_visualizations(results)
    figure('Name', 'Comparativo da Razão Teta/Beta (FFT vs CWT)', 'Position', [100, 100, 1000, 500]);
    
    % Boxplot para FFT
    subplot(1, 2, 1);
    all_fft_data = [];
    fft_groups = [];
    for g = 1:length(results.fft.tb_ratios_by_group)
        data = results.fft.tb_ratios_by_group{g};
        all_fft_data = [all_fft_data; data];
        fft_groups = [fft_groups; repmat(results.group_labels(g), length(data), 1)];
    end
    
    if ~isempty(all_fft_data)
        boxplot(all_fft_data, fft_groups);
        title('Análise com FFT');
        ylabel('Razão Teta/Beta');
        grid on;
        ylim([0, max(all_fft_data)*1.1]);
    else
        title('Análise com FFT - Sem Dados');
        text(0.5, 0.5, 'Nenhum dado válido foi gerado.', 'HorizontalAlignment', 'center', 'FontSize', 10);
        set(gca, 'XTick', [], 'YTick', []);
    end

    % Boxplot para CWT
    subplot(1, 2, 2);
    all_cwt_data = [];
    cwt_groups = [];
    for g = 1:length(results.cwt.tb_ratios_by_group)
        data = results.cwt.tb_ratios_by_group{g};
        all_cwt_data = [all_cwt_data; data];
        cwt_groups = [cwt_groups; repmat(results.group_labels(g), length(data), 1)];
    end
    
    if ~isempty(all_cwt_data)
        boxplot(all_cwt_data, cwt_groups);
        title('Análise com Wavelet (CWT)');
        ylabel('Razão Teta/Beta');
        grid on;
        ylim([0, max(all_cwt_data) * 1.1]);
    else
        title('Análise com Wavelet (CWT) - Sem Dados');
        text(0.5, 0.5, 'Nenhum dado válido foi gerado.', 'HorizontalAlignment', 'center', 'FontSize', 10);
        set(gca, 'XTick', [], 'YTick', []);
    end
    
    sgtitle('Comparativo de Resultados entre Métodos de Análise');
end

% --- NOVO ---
function create_spectrogram_visualization(all_data, fs)
    % Pega o primeiro sujeito de um grupo TDAH e um grupo Controle para exemplo
    % (Assumindo que o primeiro grupo é ADHD e o segundo é Controle)
    if ~isempty(all_data{1}) && ~isempty(all_data{2})
        signal_adhd = all_data{1}{1}(1,:); % Primeiro sujeito TDAH, primeiro canal
        signal_ctrl = all_data{2}{1}(1,:); % Primeiro sujeito Controle, primeiro canal

        figure('Name', 'Análise Tempo-Frequência (Espectrograma)', 'Position', [150, 150, 1000, 500]);

        % Espectrograma para TDAH
        subplot(1, 2, 1);
        win = hamming(fs); % Janela de 1 segundo
        spectrogram(signal_adhd, win, length(win)/2, fs, fs, 'yaxis');
        title('Espectrograma de Exemplo (TDAH)');
        
        % Espectrograma para Controle
        subplot(1, 2, 2);
        spectrogram(signal_ctrl, win, length(win)/2, fs, fs, 'yaxis');
        title('Espectrograma de Exemplo (Controle)');
        
        sgtitle('Comparativo de Espectrogramas');
    else
        fprintf('Não foi possível gerar espectrogramas: dados de exemplo não encontrados.\n');
    end
end
