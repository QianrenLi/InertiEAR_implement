clc
clear
test = 1:55;
file_str = cell(1,55);
for i = test
   file_str{i} = num2str(i,'%02d');
end

%% Create File
% target_file_str = cell(1,55);
% for i = 1:55
%     target_file_str{i} = sprintf("./data10/%s",file_str{i});
%     mkdir(target_file_str{i})
% end
%% Change intensity
target_file_str = cell(1,55);
source_file_str = cell(1,55);
% Normalized / 10
factor =10;
for i = 1:55
    source_file_str{i} = sprintf("./data/%s",file_str{i});
    target_file_str{i} = sprintf("./data10/%s",file_str{i});
end
%%
for i = 1:55
    
    file_names = dir(fullfile(source_file_str{i},'*.wav'));
    for j = 1: length(file_names)
        % for i = 1:length(file_names)
        file_name = sprintf("%s/%s",source_file_str{i},file_names(j).name);
        target_file_name = sprintf("%s/%s",target_file_str{i},file_names(j).name);
        [temp_wave,Fs] = audioread(file_name);

        % Normalized
%         temp_wave = normalize(temp_wave)/factor;
        % Simply multiply
        temp_wave = temp_wave * factor;
        audiowrite(target_file_name,temp_wave,Fs)
    end
    fprintf("Finish %d file\n",i);
end
