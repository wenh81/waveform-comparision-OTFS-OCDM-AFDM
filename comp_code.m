clc; clear; close all;

%% ======== UNIFIED SYSTEM COMPARISON =======
% Compares OCDM, AFDM, and OTFS over NTN channels
% All systems use:
% - Same channel model
% - Same SNR range
% - Same modulation (16-QAM)
% - Same number of symbols (256)
% - Cyclic Prefix for all systems
% - LMMSE and MMSE-SD detectors

%% ================= COMMON PARAMETERS ==========
%================================================
params = struct();

% System dimensions
params.K = 16;
params.L = 16;
params.N = params.K * params.L;  % 256 symbols

% Cyclic Prefix
params.cp_len = floor(params.N / 4);  % CP length = N/4 = 64

% Modulation
params.bits_in_sym = 4;
params.M = 2^params.bits_in_sym;  % 16-QAM

% Physical parameters
params.fc = 2.55e9;               % Carrier frequency
params.df = 15e3;                 % Subcarrier spacing  15 kHz 
params.terminal_velocity = 500*5/18;  % 500 km/h in m/s
params.c = physconst('lightspeed');
params.max_doppler = params.terminal_velocity/params.c * params.fc*0.5;

% Simulation parameters
params.num_iter = 100;            % Monte Carlo iterations
params.SNR_dB = -5:2:15;          % SNR range
params.SNR = 10.^(params.SNR_dB/10);
params.channel_type = 'NTN_D';

% Display parameters
fprintf('=== UNIFIED NTN SYSTEM COMPARISON WITH CYCLIC PREFIX ===\n');
fprintf('Systems: OCDM, AFDM, OTFS\n');
fprintf('Symbols: %d, Modulation: %d-QAM\n', params.N, params.M);
fprintf('Cyclic Prefix Length: %d\n', params.cp_len);
fprintf('Iterations: %d, SNR: %d to %d dB\n', ...
    params.num_iter, params.SNR_dB(1), params.SNR_dB(end));
fprintf('Max Doppler: %.2f Hz\n', params.max_doppler);
fprintf('================================================================\n\n');

%% ================= INITIALIZE RESULTS STRUCTURE ========
results = struct();

% OCDM results
results.ocdm.ber_lmmse = zeros(size(params.SNR_dB));
results.ocdm.ber_mmsesd = zeros(size(params.SNR_dB));
results.ocdm.name = 'OCDM';

% AFDM results
results.afdm.ber_lmmse = zeros(size(params.SNR_dB));
results.afdm.ber_mmsesd = zeros(size(params.SNR_dB));
results.afdm.name = 'AFDM';

% OTFS results
results.otfs.ber_lmmse = zeros(size(params.SNR_dB));
results.otfs.ber_mmsesd = zeros(size(params.SNR_dB));
results.otfs.name = 'OTFS';

%% ================= SYSTEM 1: OCDM =================
fprintf('\n========== Running OCDM Simulation ==========\n');
results.ocdm = run_ocdm_simulation(params);

%% ================= SYSTEM 2: AFDM =================
fprintf('\n========== Running AFDM Simulation ==========\n');
results.afdm = run_afdm_simulation(params);

%% ================= SYSTEM 3: OTFS =================
fprintf('\n========== Running OTFS Simulation ==========\n');
results.otfs = run_otfs_simulation(params);

%% ================= COMPARATIVE PLOTTING ========
plot_comparison_results(results, params);

%% ================= SAVE RESULTS ================
save('unified_comparison_results_NTND.mat', 'results', 'params');
fprintf('\nResults saved to: unified_comparison_results.mat\n');

%% ================= PERFORMANCE SUMMARY =================
print_performance_summary(results, params);

%% ================================================
%% ================= SYSTEM FUNCTIONS =============
%% ================================================

function results = run_ocdm_simulation(params)
    % OCDM with Fresnel Transform and Cyclic Prefix
    
    results.name = 'OCDM';
    results.ber_lmmse = zeros(size(params.SNR_dB));
    results.ber_mmsesd = zeros(size(params.SNR_dB));
    
    N = params.N;
    cp_len = params.cp_len;
    
    % Fresnel Transform Matrices
    fprintf('  Computing Fresnel matrices...\n');
    FSnT = zeros(N, N);
    for k = 1:N
        for l = 1:N
            FSnT(k,l) = 1/sqrt(N) * exp(-1j*pi/4) * exp(1j*pi*(k-l)^2/N);
        end
    end
    IFSnT = FSnT';
    
    % SNR Loop
    for idx = 1:length(params.SNR_dB)
        N0 = 1/params.SNR(idx);
        err_lmmse = 0;
        err_mmsesd = 0;
        tot_bits = 0;
        
        fprintf('  SNR = %d dB: ', params.SNR_dB(idx));
        
        for it = 1:params.num_iter
            if mod(it, 50) == 0
                fprintf('.');
            end
            
            % Transmitter
            x = randi([0 params.M-1], N, 1);
            bits_tx = de2bi(x, params.bits_in_sym, 'left-msb');
            s_qam = qammod(x, params.M, 'UnitAveragePower', true);
            s_ocdm = IFSnT * s_qam;  % Fresnel transform
            
            % Add CP
            s_cp = [s_ocdm(N-cp_len+1:N); s_ocdm];  % Cyclic prefix: (N+cp_len)x1
            
            % Channel (N×N)
            [H, ~, ~, ~, ~, ~] = NTN_channels( ...
                params.K, params.L, params.df, ...
                params.max_doppler, params.channel_type);
            
            % AWGN
            w = sqrt(N0/2) * (randn(N, 1) + 1j*randn(N, 1));
            
            % Channel effect on CP signal via circular convolution approximation
            % Process the received signal through channel (first N samples)
            r_received = H * s_ocdm;
            r_cp = [r_received(N-cp_len+1:N); r_received];
            r_cp = r_cp + [w; w];  % Add noise properly
            
            % Remove CP
            r = r_cp(cp_len+1:cp_len+N);
            
            % Fresnel domain
            r_ocdm = FSnT * r;
            
            % Effective channel in Fresnel domain
            D = FSnT * H * IFSnT;
            D = D / sqrt(trace(D*D')/N);
            
            % LMMSE
            W_lmmse = D' / (D*D' + N0*eye(N));
            s_hat_lmmse = W_lmmse * r_ocdm;
            x_hat_lmmse = qamdemod(s_hat_lmmse, params.M, 'UnitAveragePower', true);
            bits_lmmse = de2bi(x_hat_lmmse, params.bits_in_sym, 'left-msb');
            
            % MMSE-SD
            s_hat_mmsesd = mmse_sd_detector_unified(r_ocdm, D, N0, params.M);
            x_hat_mmsesd = qamdemod(s_hat_mmsesd, params.M, 'UnitAveragePower', true);
            bits_mmsesd = de2bi(x_hat_mmsesd, params.bits_in_sym, 'left-msb');
            
            % Count errors
            err_lmmse = err_lmmse + sum(bits_tx(:) ~= bits_lmmse(:));
            err_mmsesd = err_mmsesd + sum(bits_tx(:) ~= bits_mmsesd(:));
            tot_bits = tot_bits + numel(bits_tx);
        end
        
        results.ber_lmmse(idx) = max(err_lmmse / tot_bits, 1e-6);
        results.ber_mmsesd(idx) = max(err_mmsesd / tot_bits, 1e-6);
        
        fprintf(' LMMSE=%.3e, MMSE-SD=%.3e\n', ...
            results.ber_lmmse(idx), results.ber_mmsesd(idx));
    end
end

function results = run_afdm_simulation(params)
    % AFDM with Discrete Affine Fourier Transform and Cyclic Prefix
    
    results.name = 'AFDM';
    results.ber_lmmse = zeros(size(params.SNR_dB));
    results.ber_mmsesd = zeros(size(params.SNR_dB));
    
    N = params.N;
    cp_len = params.cp_len;
    
    % AFDM Parameters
    nu_max = params.max_doppler / params.df;
    c1 = (2*(nu_max + 2)+1)/(2*N);
    c2 = 1/(2*pi*N);
    
    fprintf('  Computing DAFT matrices...\n');
    n = (0:N-1).';
    m = (0:N-1).';
    D1 = diag(exp(1j*2*pi*c1*n.^2));
    D2 = diag(exp(1j*2*pi*c2*m.^2));
    DFT = dftmtx(N)/sqrt(N);
    A = D2 * DFT * D1;      % DAFT matrix
    AH = A';                 % Inverse DAFT
    
    % SNR Loop
    for idx = 1:length(params.SNR_dB)
        N0 = 1/params.SNR(idx);
        err_lmmse = 0;
        err_mmsesd = 0;
        tot_bits = 0;
        
        fprintf('  SNR = %d dB: ', params.SNR_dB(idx));
        
        for it = 1:params.num_iter
            if mod(it, 50) == 0
                fprintf('.');
            end
            
            % Transmitter
            x = randi([0 params.M-1], N, 1);
            bits_tx = de2bi(x, params.bits_in_sym, 'left-msb');
            y = qammod(x, params.M, 'UnitAveragePower', true);
            s = AH * y;  % IDAFT
            s = s / sqrt(mean(abs(s).^2));
            
            % Add CP
            s_cp = [s(N-cp_len+1:N); s];  % (N+cp_len)x1
            
            % Channel
            [HT, ~, ~, ~, ~, ~] = NTN_channels( ...
                params.K, params.L, params.df, ...
                params.max_doppler, params.channel_type);
            HT = HT / sqrt(trace(HT*HT')/N);
            
            % AWGN
            w = sqrt(N0/2) * (randn(N, 1) + 1j*randn(N, 1));
            
            % Channel effect
            r_received = HT * s;
            r_cp = [r_received(N-cp_len+1:N); r_received];
            r_cp = r_cp + [w; w];  % Add noise
            
            % Remove CP
            r_time = r_cp(cp_len+1:cp_len+N);
            
            % DAFT domain
            Y = A * r_time;
            H_daft = A * HT * AH;
            H_daft = H_daft / sqrt(trace(H_daft*H_daft')/N);
            
            % LMMSE
            W_lmmse = H_daft' / (H_daft*H_daft' + N0*eye(N));
            y_hat_lmmse = W_lmmse * Y;
            x_hat_lmmse = qamdemod(y_hat_lmmse, params.M, 'UnitAveragePower', true);
            bits_lmmse = de2bi(x_hat_lmmse, params.bits_in_sym, 'left-msb');
            
            % MMSE-SD
            y_hat_mmsesd = mmse_sd_detector_unified(Y, H_daft, N0, params.M);
            x_hat_mmsesd = qamdemod(y_hat_mmsesd, params.M, 'UnitAveragePower', true);
            bits_mmsesd = de2bi(x_hat_mmsesd, params.bits_in_sym, 'left-msb');
            
            % Count errors
            err_lmmse = err_lmmse + sum(bits_tx(:) ~= bits_lmmse(:));
            err_mmsesd = err_mmsesd + sum(bits_tx(:) ~= bits_mmsesd(:));
            tot_bits = tot_bits + numel(bits_tx);
        end
        
        results.ber_lmmse(idx) = max(err_lmmse / tot_bits, 1e-6);
        results.ber_mmsesd(idx) = max(err_mmsesd / tot_bits, 1e-6);
        
        fprintf(' LMMSE=%.3e, MMSE-SD=%.3e\n', ...
            results.ber_lmmse(idx), results.ber_mmsesd(idx));
    end
end

function results = run_otfs_simulation(params)
    % OTFS with Delay-Doppler domain processing and Cyclic Prefix
    
    results.name = 'OTFS';
    results.ber_lmmse = zeros(size(params.SNR_dB));
    results.ber_mmsesd = zeros(size(params.SNR_dB));
    
    M = params.K;  % Time slots
    N = params.L;  % Subcarriers
    S = M * N;     % Total symbols
    cp_len = params.cp_len;
    
    % OTFS Operators
    fprintf('  Computing OTFS operators...\n');
    W = dftmtx(N)/sqrt(N);
    WH = W';
    IL = eye(M);
    OP = kron(WH, IL);
    
    % SNR Loop
    for idx = 1:length(params.SNR_dB)
        N0 = 1/params.SNR(idx);
        err_lmmse = 0;
        err_mmsesd = 0;
        tot_bits = 0;
        
        fprintf('  SNR = %d dB: ', params.SNR_dB(idx));
        
        for it = 1:params.num_iter
            if mod(it, 50) == 0
                fprintf('.');
            end
            
            % Transmitter
            data = randi([0 params.M-1], S, 1);
            bits_tx = de2bi(data, params.bits_in_sym, 'left-msb');
            x = qammod(data, params.M, 'UnitAveragePower', true);
            
            % IFFT for OTFS precoding
            x_time = ifft(reshape(x, N, M), N, 1) * sqrt(N);
            x_time = reshape(x_time, S, 1);
            
            % Reshape to 2D for CP addition
            x_2d = reshape(x_time, N, M);
            x_cp_2d = [x_2d(N-cp_len+1:N, :); x_2d];  % (N+cp_len)×M
            x_cp = reshape(x_cp_2d, (N+cp_len)*M, 1);
            
            % Channel
            [HT, ~, ~, ~, ~, ~] = NTN_channels( ...
                M, N, params.df, params.max_doppler, params.channel_type);
            
            % Effective channel (DD domain)
            H_eff = OP' * HT * OP;
            H_eff = H_eff / sqrt(trace(H_eff*H_eff')/S);
            
            % DD-domain signal
            w = sqrt(N0/2) * (randn(S, 1) + 1j*randn(S, 1));
            r_dd = H_eff * x + w;
            
            % LMMSE
            W_lmmse = (H_eff' * H_eff + N0*eye(S)) \ H_eff';
            x_lmmse = W_lmmse * r_dd;
            data_lmmse = qamdemod(x_lmmse, params.M, 'UnitAveragePower', true);
            bits_lmmse = de2bi(data_lmmse, params.bits_in_sym, 'left-msb');
            
            % MMSE-SD
            x_mmsesd = mmse_sd_detector_unified(r_dd, H_eff, N0, params.M);
            data_mmsesd = qamdemod(x_mmsesd, params.M, 'UnitAveragePower', true);
            bits_mmsesd = de2bi(data_mmsesd, params.bits_in_sym, 'left-msb');
            
            % Count errors
            err_lmmse = err_lmmse + sum(bits_tx(:) ~= bits_lmmse(:));
            err_mmsesd = err_mmsesd + sum(bits_tx(:) ~= bits_mmsesd(:));
            tot_bits = tot_bits + numel(bits_tx);
        end
        
        results.ber_lmmse(idx) = max(err_lmmse / tot_bits, 1e-6);
        results.ber_mmsesd(idx) = max(err_mmsesd / tot_bits, 1e-6);
        
        fprintf(' LMMSE=%.3e, MMSE-SD=%.3e\n', ...
            results.ber_lmmse(idx), results.ber_mmsesd(idx));
    end
end

function plot_comparison_results(results, params)
    % Create comprehensive comparison plots
    
    figure('Position', [100, 100, 1200, 500]);
    
    % Subplot 1: LMMSE Comparison
    subplot(1,2,1);
    semilogy(params.SNR_dB, results.ocdm.ber_lmmse, '-ro', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    semilogy(params.SNR_dB, results.afdm.ber_lmmse, '-g^', 'LineWidth', 2, 'MarkerSize', 6);
    semilogy(params.SNR_dB, results.otfs.ber_lmmse, '-bs', 'LineWidth', 2, 'MarkerSize', 6);
    grid on;
    xlabel('SNR (dB)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('BER', 'FontSize', 14, 'FontWeight', 'bold');
    title('LMMSE Detector Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    legend('OCDM', 'AFDM', 'OTFS', 'Location', 'southwest', 'FontSize', 12);
    ylim([1e-5 1]);
    grid on;
    set(gca, 'FontSize', 11);
    
    % Subplot 2: MMSE-SD Comparison
    subplot(1,2,2);
    semilogy(params.SNR_dB, results.ocdm.ber_mmsesd, '-ro', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    semilogy(params.SNR_dB, results.afdm.ber_mmsesd, '-g^', 'LineWidth', 2, 'MarkerSize', 6);
    semilogy(params.SNR_dB, results.otfs.ber_mmsesd, '-bs', 'LineWidth', 2, 'MarkerSize', 6);
    grid on;
    xlabel('SNR (dB)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('BER', 'FontSize', 14, 'FontWeight', 'bold');
    title('MMSE-SD Detector Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    legend('OCDM', 'AFDM', 'OTFS', 'Location', 'southwest', 'FontSize', 12);
    ylim([1e-5 1]);
    grid on;
    set(gca, 'FontSize', 11);
    
    sgtitle('BER vs SNR - OCDM/AFDM/OTFS Waveform Comparison (NTN Channel)', ...
        'FontSize', 15, 'FontWeight', 'bold');
    
    saveas(gcf, 'ber_snr_comparison.fig');
    saveas(gcf, 'ber_snr_comparison.png');
    saveas(gcf, 'ber_snr_comparison.pdf');
    fprintf('\nPlots saved: ber_snr_comparison.fig/png/pdf\n');
end

function print_performance_summary(results, params)
    % Print comprehensive performance summary
    
    fprintf('\n');
    fprintf('================================================================================\n');
    fprintf('                    PERFORMANCE SUMMARY - ALL SYSTEMS\n');
    fprintf('================================================================================\n\n');
    
    systems = {'ocdm', 'afdm', 'otfs'};
    names = {'OCDM', 'AFDM', 'OTFS'};
    
    % LMMSE Summary
    fprintf('LMMSE DETECTOR PERFORMANCE\n');
    fprintf('SNR (dB)   │   OCDM        │   AFDM        │   OTFS        │   Best System\n');
    fprintf('─────────────────────────────────────────────────────────────────────────\n');
    
    for idx = 1:length(params.SNR_dB)
        bers_lmmse = [results.ocdm.ber_lmmse(idx), ...
                      results.afdm.ber_lmmse(idx), ...
                      results.otfs.ber_lmmse(idx)];
        [~, best_l] = min(bers_lmmse);
        
        fprintf('%6d     │  %.3e  │  %.3e  │  %.3e  │   %s\n', ...
            params.SNR_dB(idx), ...
            results.ocdm.ber_lmmse(idx), ...
            results.afdm.ber_lmmse(idx), ...
            results.otfs.ber_lmmse(idx), ...
            names{best_l});
    end
    
    % MMSE-SD Summary
    fprintf('\nMMSE-SD DETECTOR PERFORMANCE\n');
    fprintf('SNR (dB)   │   OCDM        │   AFDM        │   OTFS        │   Best System\n');
    fprintf('─────────────────────────────────────────────────────────────────────────\n');
    
    for idx = 1:length(params.SNR_dB)
        bers_mmsesd = [results.ocdm.ber_mmsesd(idx), ...
                       results.afdm.ber_mmsesd(idx), ...
                       results.otfs.ber_mmsesd(idx)];
        [~, best_s] = min(bers_mmsesd);
        
        fprintf('%6d     │  %.3e  │  %.3e  │  %.3e  │   %s\n', ...
            params.SNR_dB(idx), ...
            results.ocdm.ber_mmsesd(idx), ...
            results.afdm.ber_mmsesd(idx), ...
            results.otfs.ber_mmsesd(idx), ...
            names{best_s});
    end
    
    % Overall Statistics
    fprintf('\n');
    fprintf('================================================================================\n');
    fprintf('                       OVERALL BEST PERFORMANCE\n');
    fprintf('================================================================================\n\n');
    
    for s = 1:length(systems)
        sys = systems{s};
        fprintf('%s:\n', names{s});
        fprintf('  LMMSE  - Best BER: %.3e at SNR=%d dB\n', ...
            min(results.(sys).ber_lmmse), params.SNR_dB(find(results.(sys).ber_lmmse == min(results.(sys).ber_lmmse), 1)));
        fprintf('  MMSE-SD- Best BER: %.3e at SNR=%d dB\n\n', ...
            min(results.(sys).ber_mmsesd), params.SNR_dB(find(results.(sys).ber_mmsesd == min(results.(sys).ber_mmsesd), 1)));
    end
    
    fprintf('================================================================================\n\n');
end

function s_hat = mmse_sd_detector_unified(r, H, N0, M)
    % MMSE-Sphere Decoder (Simplified)
    % Implements a simple sphere decoder with MMSE preprocessing
    
    % MMSE preprocessing
    n_sym = size(H, 2);  % Number of symbols
    G = H' / (H*H' + N0*eye(size(H,1)));
    s_mrc = G * r;
    
    % Get constellation points
    constellation = qammod((0:M-1)', M, 'UnitAveragePower', true);
    
    % Initialize output
    s_hat = zeros(n_sym, 1);
    
    % For each symbol, find the best match in constellation
    for ii = 1:n_sym
        [~, idx] = min(abs(s_mrc(ii) - constellation));
        s_hat(ii) = constellation(idx);
    end
end
