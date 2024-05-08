% Load the ECG data from the .mat file
load('ECGData.mat'); % Replace 'your_ecg_data_file.mat' with your actual file name
data = ECGData.Data;
lables = ECGData.Labels;
% Assuming your ECG data is stored in a variable named ecg_data
% If your variable name is different, replace 'ecg_data' with your actual variable name

% Plot the ECG signal
figure;
plot(data);
title('ECG Signal');
xlabel('Time (samples)');
ylabel('Amplitude');
grid on;

% If you have the sampling frequency information, you can set the x-axis in seconds
% For example, if your sampling frequency is fs Hz, you can use:
% time_seconds = (0:length(ecg_data)-1) / fs;
% plot(time_seconds, ecg_data);
% xlabel('Time (seconds)');

% Customize the plot as needed, such as adding labels, grid, etc.