clear
close all
clc

fs = 360;                % sampling frequency (samples per second/Hz) 
dt = 1/fs;               % seconds per sample 
stopTime = 10;            % length of signal in seconds 
t = (0:dt:stopTime-dt); % time vector in seconds 
stopTime_plot = 2;       % limit time axis for improved visualization 




% Load the data
load('300m.mat');
y = val;
load('bwm.mat');
n_s = val;

% Normalize function to scale between -1 and 1
%normalize_signal =  2 * (x - min(x)) / (max(x) - min(x)) - 1;

% Normalize the signals
%y = normalize_signal(y);
%n_s = normalize_signal(n_s);

% EEMD parameters
noise_std = 0.2;         % 20% of the standard deviation of the signal
ensemble_count = 100;    % Number of ensembles

% Create the noisy signal
y_noise = y + n_s;
%y_noise =  2 * (y_noise - min(y_noise)) / (max(y_noise) - min(y_noise)) - 1;
%y_noise = normalize_signal(y_noise);  % Normalize the noisy signal
%y_3=y_noise';
%y =  2 * (y- min(y_noise)) / (max(y_noise) - min(y_noise)) - 1;
%n_s =  2 * (n_s - min(y_noise)) / (max(y_noise) - min(y_noise)) - 1;

input_signal_power = rms(y).^2;  % Power of the input signal
input_noise_power = rms(n_s).^2;    % Power of the input noise
input_snr = 10 * log10(input_signal_power / input_noise_power);  % Input SNR in dB
disp(['Signal-to-Noise Ratio (SNR): ', num2str(input_snr), ' dB']);

% Using built-in functions if you have the toolbox
mse_value = immse(y, y_noise);
% Load the sample data

% Display the result
disp(['Mean Squared Error (MSE) using immse: ', num2str(mse_value)]);
% Perform EMD on the noisy signal

% Initialize variables
z= length(y);
%imfs = zeros( ensemble_count,z,8);  % To store IMFs of each ensemble
modes=zeros(z,8);
imfs = [];

% Perform EEMD
for i = 1:ensemble_count
    % Add white noise to the signal
    noise = noise_std * randn(size(y));
    noisy_signal = y_noise+ noise;
    
    % Perform EMD on the noisy signal
    imf_temp = emd(noisy_signal, 'MaxNumIMF', 14);
    modes= imf_temp;
    % Display the number of IMFs
num_imfs = size(imf_temp, 2);
fprintf('Number of IMFs: %d\n', num_imfs);
    
    % Store the IMFs
    imfs(i,:, 1:size(imf_temp, 2)) = imf_temp;
    %imfs(i,:, :) = imf_temp;
    % Store the IMFs
   % imfs(i,:,:) = imf_temp;
   %imfs(i, :, :) = modes;
end

% Average the IMFs across ensembles

% Average the IMFs across ensembles (1st dimension)
avg_imfs = squeeze(mean(imfs, 1));

% Display the number of IMFs
num_imfs = size(avg_imfs, 2);
fprintf('Number of IMFs: %d\n', num_imfs);

g=size(avg_imfs,2);
    x = zeros(1,length(y));
%m = round(0.24 * num_imfs); 
imf_2= avg_imfs';
for i = 1:g-4
    x = x + imf_2(i,:);
end   

x_noise=y_noise-x;
%x_noi=x_noise';
% Opening operation on x signal (example of a 3-point opening)
se1 = strel('line', 0.11*fs, 0);  % Create a structuring element (line, length 3)
x_o1 = imopen(x, se1); % Perform opening operation
% Opening operation on x signal (example of a 3-point opening)
  % Create a structuring element (line, length 3)
x_c1= imclose(x, se1); % Perform opening operation
 
 xmp1=(x_o1+x_c1)/2;
 %=xmp1-x_n1;
% Opening operation on x signal (example of a 3-point opening)
se2= strel('line', 0.3*fs, 0);  % Create a structuring element (line, length 3)
x_o2 = imopen(xmp1, se2); % Perform opening operation
% Opening operation on x signal (example of a 3-point opening)
%se4 = strel('line', 0.11*fs, 0);  % Create a structuring element (line, length 3)
x_c2= imclose(xmp1, se2); % Perform opening operation
xmp2=(x_o2+x_c2)/2;

y_f2=x-xmp2;



num=0;
den=0;
z=0;
for i=1:length(y)
    den=den+(y(i)-x(i))^2;
end

for i=1:length(y)
    num=num+y(i)^2;
end


SNR=20*log10(sqrt(num)/sqrt(den));

disp(['Signal-to-Noise Ratio (SNR): ', num2str(SNR), ' dB']);
mse_out = immse(y, x);
disp(['Mean Squared Error (MSE) using immse: ', num2str(mse_out)]);
figure;
subplot(2, 1, 1);
plot(t, y);
title('Original ECG Signal');

subplot(2, 1, 2);
plot(t, y_f2);
title('Filtured  ECG Signal');
figure;
subplot(6,1,1);
plot(t,x);
title('x_noise');
xlabel('Time (samples)');
ylabel('Amplitude');

subplot(6,1,2);
plot(t,xmp1);
title('xmp1');
xlabel('Time (samples)');
ylabel('Amplitude');

subplot(6,1,3);
plot(t,xmp2);
title('xmp2');
xlabel('Time (samples)');
ylabel('Amplitude');

subplot(6,1,4);
plot(t,y_f2);
title(' xecg1  signal');
xlabel('Time (samples)');
ylabel('Amplitude');

subplot(6,1,5);
plot(t,n_s);
title('noise signal');
xlabel('Time (samples)');
ylabel('Amplitude');

subplot(6,1,6);
plot(t,x_noise);
title('baseline signal');
xlabel('Time (samples)');
ylabel('Amplitude');

figure;
plot(t,x);
title('x_noise');
xlabel('Time (samples)');
ylabel('Amplitude');