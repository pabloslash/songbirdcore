clear; clc; close all


fs = 25000;
lowcut = 300;
highcut = 8000;
order = 4;

% Band-pass filter
[b,a] = butter(order, [lowcut,highcut]/(fs/2),'bandpass');

% IIR high-pass filter
[b2,a2] = butter(order, lowcut/(fs/2), 'high');

% FIR high-pass filter
b3 = fir1(1000, lowcut/(fs/2), 'high');

save('filters/butter_bp_300Hz-8000hz_order4_sr25000.mat','a','b'); 

    
%% Filter singal

%load('song_neural_ex.mat')
% waveform = - waveform;

filtIIR = filter(b2, a2, waveform);
filtFIR = filter(b3, 1, waveform);
filtBP = filter(b, a, waveform);

filtfiltIIR = filtfilt(b2, a2, waveform);
filtfiltFIR = filtfilt(b3, 1, waveform);
filtfiltBP = filtfilt(b, a, waveform);

%% Visualize Filter
[h, w] = freqz(b2,a2,fs,fs);
[h2, w2] = freqz(b3,1,fs,fs); 
[h3, w3] = freqz(b,a,fs,fs); 

% Filter
figure()

subplot(2, 1, 1)
semilogx(w,20*log10(abs(h)), 'k', 'LineWidth', 2); hold on;
semilogx(w2,20*log10(abs(h2)), 'r--', 'LineWidth', 2);
semilogx(w3,20*log10(abs(h3)), 'b', 'LineWidth', 2);
xlim([0 12000])
title('Filter Response', 'FontSize', 20)
xlabel('Freq (Hz)')
ylabel('Magnitude Response (dB)')
legend(['Butter high: order ', num2str(order)], 'FIR: order 1000', ['Butter band: order ', num2str(order)],'Location', 'Best')
set(gcf,'color', 'white')

subplot(2, 1, 2)
semilogx(w,unwrap(angle(h)*180/pi), 'k--', 'LineWidth', 2); hold on;
semilogx(w2,unwrap(angle(h2)*180/pi), 'r--', 'LineWidth', 2);
semilogx(w3,unwrap(angle(h3)*180/pi), 'b--', 'LineWidth', 2);
xlim([0 12000])
xlabel('Freq (Hz)')
ylabel('Phase Response (degrees)')
legend('Butter high', 'FIR', 'Butter band','Location', 'Best')
set(gcf,'color', 'white')

%% Bandpass Filter

figure()

subplot(2, 1, 1)
semilogx(w3,20*log10(abs(h3)), 'b', 'LineWidth', 2);
xlim([0 20000])
title('Filter Response', 'FontSize', 20)
xlabel('Freq (Hz)')
ylabel('Magnitude Response (dB)')
legend(['Butter band: order ', num2str(order)],'Location', 'Best')
set(gcf,'color', 'white')

subplot(2, 1, 2)
semilogx(w3,unwrap(angle(h3)*180/pi), 'b--', 'LineWidth', 2);
xlim([0 20000])
xlabel('Freq (Hz)')
ylabel('Phase Response (degrees)')
legend('Butter band','Location', 'Best')
set(gcf,'color', 'white')

%% Plot Signals

pTime = 10; %seconds
samples = pTime * fs;

figure()
plot(linspace(0, samples/fs, samples), waveform(1:samples)); hold on;
plot(linspace(0, samples/fs, samples), filtIIR(1:samples))
plot(linspace(0, samples/fs, samples), filtFIR(1:samples))
plot(linspace(0, samples/fs, samples), filtBP(1:samples))
legend('Raw', 'Butter high', 'FIR', 'Butter band')
title('Filter Signal', 'FontSize', 20)
set(gcf, 'color', 'white')
xlabel('Time (s)', 'FontSize', 18)
ylabel('Volts \mu V', 'FontSize', 18)


figure()
plot(linspace(0, samples/fs, samples), waveform(1:samples)); hold on;
plot(linspace(0, samples/fs, samples), filtfiltIIR(1:samples))
plot(linspace(0, samples/fs, samples), filtfiltFIR(1:samples))
plot(linspace(0, samples/fs, samples), filtfiltBP(1:samples))
legend('Raw', 'Butter high', 'FIR', 'Butter band')
title('FiltFilt Signal', 'FontSize', 20)
set(gcf, 'color', 'white')
xlabel('Time (s)', 'FontSize', 18)
ylabel('Volts \mu V', 'FontSize', 18)


%% Threshold Crossings

th_min = 3.5; % Threshold * SD
th_max = 120; % Threshold * SD

% RMS & threshold of filtered signal
RMSiir = sqrt(mean(filtfiltIIR.^2));
RMSfir = sqrt(mean(filtfiltFIR.^2));
RMSbp = sqrt(mean(filtfiltBP.^2));

th_iir_min = th_min * RMSiir;
th_fir_min = th_min * RMSfir;
th_bp_min = th_min * RMSbp;

th_iir_max = th_max * RMSiir;
th_fir_max = th_max * RMSfir;
th_bp_max = th_max * RMSbp;

% Indexes of samples below threshold
spikes_iir = find(filtfiltIIR > th_iir_min & filtfiltIIR < th_iir_max);
spikes_fir = find(filtfiltFIR > th_fir_min  & filtfiltFIR < th_fir_max);
spikes_bp = find(filtfiltBP > th_bp_min  & filtfiltBP < th_bp_max);

% Clean Spike detection
timeout = (1 / 1000) * fs; % 1ms timeout where 2 spikes cannot occur

spikes_iir_clean = [];
current_spike = -1000;
for i = 1:length(spikes_iir)
    if spikes_iir(i) > (current_spike+timeout)
        spikes_iir_clean = [spikes_iir_clean spikes_iir(i)];
        current_spike = spikes_iir(i);
    end
end

spikes_fir_clean = [];
current_spike = -1000;
for i = 1:length(spikes_fir)
    if spikes_fir(i) > (current_spike+timeout)
        spikes_fir_clean = [spikes_fir_clean spikes_iir(i)];
        current_spike = spikes_fir(i);
    end
end

spikes_bp_clean = [];
current_spike = -1000;
for i = 1:length(spikes_bp)
    if spikes_bp(i) > (current_spike+timeout)
        spikes_bp_clean = [spikes_bp_clean spikes_bp(i)];
        current_spike = spikes_bp(i);
    end
end


%% Plot Snippets of Threshold Crossings

t_before = 0.2; %ms
t_AP =     0.5; 
t_after =  0.2; 

s_before = t_before/1000*fs; % In samples
s_AP = t_AP/1000*fs; 
s_after = t_after/1000*fs;  

% Delete spikes that do not have enough samples before / after to capture the whole depolarization
spikes_iir_clean = spikes_iir_clean((spikes_iir_clean>=s_before) & (spikes_iir_clean<length(filtIIR) + s_AP + s_after));
spikes_fir_clean = spikes_fir_clean((spikes_fir_clean>=s_before) & (spikes_fir_clean<length(filtIIR) + s_AP + s_after));
spikes_bp_clean = spikes_bp_clean((spikes_bp_clean>=s_before) & (spikes_bp_clean<length(filtIIR) + s_AP + s_after));


% Take AP snippets
AP_iir = zeros(length(spikes_iir_clean), s_before + s_AP + s_after + 1);
AP_fir = zeros(length(spikes_fir_clean), s_before + s_AP + s_after + 1);
AP_bp = zeros(length(spikes_bp_clean), s_before + s_AP + s_after + 1);

for ap = 1: length(spikes_iir_clean)
    AP_iir(ap,:) = filtIIR(spikes_iir_clean(ap)-s_before : spikes_iir_clean(ap) + s_AP + s_after)';
end

for ap = 1: length(spikes_fir_clean)
    AP_fir(ap,:) = filtFIR(spikes_fir_clean(ap)-s_before : spikes_fir_clean(ap) + s_AP + s_after)';
end

for ap = 1: length(spikes_bp_clean)
    AP_bp(ap,:) = filtBP(spikes_bp_clean(ap)-s_before : spikes_bp_clean(ap) + s_AP + s_after)';
end

figure()
plot(linspace(0, t_before+t_AP+t_after, s_before+s_AP+s_after+1), AP_iir')
title(['IIR: ', num2str(size(AP_iir,1)), ' Spikes'])
set(gcf, 'color', 'white')
xlabel('Time (s)')
ylabel('Volts \muV')

figure()
plot(linspace(0, t_before+t_AP+t_after, s_before+s_AP+s_after+1), AP_fir')
title(['FIR: ', num2str(size(AP_fir,1)), ' Spikes'])
set(gcf, 'color', 'white')
xlabel('Time (s)')
ylabel('Volts \muV')

figure()
plot(linspace(0, t_before+t_AP+t_after, s_before+s_AP+s_after+1), AP_bp')
title(['BP: ', num2str(size(AP_bp,1)), ' Spikes'])
set(gcf, 'color', 'white')
xlabel('Time (s)')
ylabel('Volts \muV')






