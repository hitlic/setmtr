data_names = ["iAB_RBC_283", "iAF692", "iAF1260b", "iHN637", "iIT341", "iJO1366"];

for i=1:size(data_names,2)
    generate_txt_data(data_names(i))
end


function generate_txt_data(data_name)
    data_name
    load(sprintf('data/%s.mat', data_name), 'Model')
    S = Model.S;    % stoichiometric matrix S, which is a signed incidence matrix 
    S = full(spones(S));   % make S binary, now is a standard incidence matrix                      真超边
    S_names = Model.rxns;   % reaction names (hyperplinks)                                          超边名

    M = Model.mets;   % metabolite names (nodes)                                                    蛋白质名

    save_data(S, S_names, M, sprintf("%s.txt", data_name));

    U = full(spones(Model.US)); % universal reactions for this model, containing all columns of S   真假超边
    U_names = Model.unrnames; % universal reactions' names                                          真假超边名


    tmp = strcmp(repmat(S_names,1,size(U_names,2)),repmat(U_names,size(S_names,1),1));  % compare train with U
    check_unrepeat = find(sum(tmp)==0);             % 不包含在train_names中的U_names
    fake = U(:,check_unrepeat);  % 不包含在S中的U即假超边
    fake_names = U_names(:,check_unrepeat);

    save_data(fake, fake_names, M, sprintf("%s_fake.txt", data_name))


    function save_data(incidence, col_names, row_names, file_name)
        % col_names: 列名（反应名，超边名或集合名）
%         % row_names: 行名（蛋白质名，元素名）
        % data: 数据文件名
        file = fopen(file_name, 'w');
        sizes = zeros(1, size(incidence, 2));
        for i = 1:size(incidence, 2)
            col = incidence(:, i);
            sizes(i) = sum(col);
            idxs = find(col>0);
            set_name = col_names(i);
            line_s = strcat(set_name{1}, ':');
            for j = 1: size(idxs)
                idx = idxs(j);
                row = row_names(idx);
                line_s = strcat(line_s, row{1}, ",");
            end
            line_s = extractBetween(line_s, 1, strlength(line_s)-1);
            fprintf(file, line_s);
            fprintf(file, '\n');
        end
        sizes = sort(sizes);
        sizes(end-10:end)
        fclose(file);

    end
end