
data = readmatrix("Moving_Justac_5m_30FPS"); 
data = data(:, 1:end-1);

% RAW 30th frame
figure;
plot(data(30, :)); hold on;

% MIT static obj rem
reference_frame = data(1, :);
mit_filtered_data = reference_frame - data;
writematrix(mit_filtered_data, 'mit-filtered-data.txt');

% MIT filtered 30th frame
plot(mit_filtered_data(30,:));

print('Raw and filtered 30th frame', '-dpng', '-r300');

% Extract envelope of 30th frame, dropping first 100 signals
slow_time_frame_1 = mit_filtered_data(30, 100:end);
[upl, lol] = envelope(slow_time_frame_1, 5, 'peak');

figure;
plot(slow_time_frame_1, 'b'); hold on;
plot(upl, 'r', 'LineWidth', 1.5);

print('Filtered and envelope of 30th frame', '-dpng', '-r300');


% Extract envelope of all frames
all_upl = [];
for i = 1:size(data)
    raw_signal_frame = mit_filtered_data(i, :);
    [upl, lol] = envelope(raw_signal_frame, 5, 'peak'); 
    all_upl = [all_upl; upl];
end

save('upper_envelopes.txt', 'all_upl', '-ascii', '-tabs');