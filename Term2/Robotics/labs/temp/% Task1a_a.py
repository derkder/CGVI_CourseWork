% Task1a_a
% 定义经纬度和高度
latitude_deg = -33.821075;   % 纬度，单位为度
longitude_deg = 151.188496; % 经度，单位为度
height = 120;               % 高度，单位为米

% 使用 deg_to_rad 变量将经纬度从度转换为弧度
latitude_rad = latitude_deg * deg_to_rad;
longitude_rad = longitude_deg * deg_to_rad;

% 设定速度为零
velocity_ned = [0; 0; 0];

% 调用 pv_NED_to_ECEF 函数
[r_eb_e, v_eb_e] = pv_NED_to_ECEF(latitude_rad, longitude_rad, height, velocity_ned);

% 显示结果
disp('ECEF 坐标：');
disp(r_eb_e);

% Task1a_b
% 读取伪距数据文件
data = csvread('Workshop1_Pseudo_ranges.csv'); % 忽略第一行和第一列的标题

% 获取时间列和卫星编号行
time = data(:, 1);
satellite_numbers = data(1, 2:end);
% disp(satellite_numbers);
satellite_positions = zeros(3, length(satellite_numbers));
% 在时间0时刻计算每颗卫星的ECEF位置
for j = satellite_numbers
    [sat_r_es_e, ~] = Satellite_position_and_velocity(0, j);
    satellite_positions(:, j) = sat_r_es_e;
    fprintf('卫星%d的ECEF位置：X=%f m, Y=%f m, Z=%f m\n', j, sat_r_es_e(1), sat_r_es_e(2), sat_r_es_e(3));
end

% Task1a_c
% 获取卫星数量
num_satellites = size(satellite_positions, 2);
% 初始化距离数组
ranges = zeros(1, num_satellites);
satellite_ranges = zeros(1, length(satellite_numbers));

% 计算每颗卫星的距离
for j = satellite_numbers
    % 计算卫星到用户的相对位置向量
    % 初始化Sagnac效应补偿矩阵为单位矩阵
    C_e_prime = eye(3);
    r_ea_e = C_e_prime * satellite_positions(:, j) - r_eb_e;
  
    % 初始距离估计
    range_old = 0;
    range_new = sqrt(r_ea_e' * r_ea_e);
    
    % 递归计算，直到新旧距离之差的绝对值小于某个阈值（例如1e-6米）
    while abs(range_new - range_old) > 1e-6
        % 更新距离估计
        range_old = range_new;
    
        % 计算Sagnac效应补偿矩阵C'_e，根据方程（2）
        C_e_prime = [1, omega_ie * range_old / c, 0;
        -omega_ie * range_old / c, 1, 0;
        0, 0, 1];
            % 应用Sagnac效应补偿矩阵来修正相对位置向量
        % r_ea_e_corrected = C_e_prime * r_ea_e;
        r_ea_e_corrected = C_e_prime * satellite_positions(:, j) - r_eb_e;
    
        % 根据修正后的向量重新计算距离
        range_new = sqrt(r_ea_e_corrected' * r_ea_e_corrected);
    end

    % 存储收敛后的距离
    satellite_ranges(j) = range_new;  
    % 打印每个卫星的预测距离
    fprintf('卫星%d的预测距离：%f m\n', j, satellite_ranges(j));
end

% Task1a_d
% Task1a_d
% 计算每颗卫星的视线单位向量
line_of_sight_vectors = zeros(3, num_satellites);

for j = satellite_numbers
    r_ea_e_corrected = C_e_prime * satellite_positions(:, j) - r_eb_e;
    line_of_sight_vectors(:, j) = r_ea_e_corrected / satellite_ranges(j);
    fprintf('卫星%d的视线单位向量：[%f, %f, %f]\n', j, line_of_sight_vectors(1, j), line_of_sight_vectors(2, j), line_of_sight_vectors(3, j));
end


% Task1a_e
% 预测状态向量和测量创新向量初始化
predicted_state_vector = zeros(4, 1);
measurement_innovation = zeros(num_satellites, 1);
H_G = zeros(num_satellites, 4);

% 假设初始预测接收机时钟偏移为0
predicted_receiver_clock_offset = 0;

% 计算预测状态向量 x̂ 和测量创新向量 δz̃
for j = satellite_numbers
    predicted_state_vector(1:3) = r_eb_e;
    predicted_state_vector(4) = predicted_receiver_clock_offset;

    % 真实测量的伪距
    rho_j = satellite_ranges(j);
    
    % 预测的伪距
    rho_j_hat = norm(satellite_positions(:, j) - r_eb_e);
    
    % 测量创新
    measurement_innovation(j) = rho_j - rho_j_hat;

    % 测量矩阵 H^e_G 的构造
    H_G(j, 1:3) = -line_of_sight_vectors(:, j)';
    H_G(j, 4) = 1;
end


% Task1a_f
% 利用无加权最小二乘计算位置和接收机时钟偏移
% 注意：这里的H_G是从e）步骤计算得到的

% 最小二乘解
delta_x = inv(H_G' * H_G) * H_G' * measurement_innovation;
estimated_position_and_clock_offset = predicted_state_vector + delta_x;

% 打印计算得到的位置和时钟偏移
disp('估计的位置和接收机时钟偏移：');
disp(estimated_position_and_clock_offset);




