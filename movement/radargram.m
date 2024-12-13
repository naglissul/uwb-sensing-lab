
M1 = readmatrix("upper_envelopes.txt");

% Radar parameters
fs = 23.328e9; % Sampling frequency in Hz
PRF = 30; % Pulse Repetition Frequency in Hz
c = 3e8; % Speed of light in m/s

% Define axes
X = (1:length(M1(1, :))) * c / (2 * fs); % Range (m)
Y = (1:size(M1, 1)) / PRF; % Slow time (s)

% Generate 2D Radargram
figure;
mesh(Y, X, M1'); % Note: Transpose M1 to align axes
xlabel('Slow time (s)');
ylabel('Range (m)');
zlabel('Amplitude');
title('2D Radargram Plot');
colormap('jet'); % Add color for better visualization
colorbar; % Add colorbar to indicate amplitude scale
view(2); % Set view to 2D

print('Radargram of all envelopes', '-dpng', '-r300');
