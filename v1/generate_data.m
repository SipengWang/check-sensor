clear
% basic line
basic = ones(1, 100);
power = -5

% add sine function
temp = 1:length(basic);
sine = 0.1 * sin(0.2 * temp);
basic = basic + sine;

% add white gauss noise
Global_noise = wgn(1, length(basic), power);
Global = basic + Global_noise;

figure;
plot(temp, Global, '*')
axis([-inf, inf, -3, 3])

% add error
GPS(1:0.5 * length(basic)) = basic(1:0.5 * length(basic));
for i = 0.5 * length(basic) + 1:length(basic)
   
    GPS(i) = basic(i) + (i - 0.5 * length(basic)) * 0.05;
    
end
GPS_noise = wgn(1, length(basic), power);
GPS = GPS + GPS_noise;

figure;
plot(1:length(basic), GPS, '*')

delta = zeros([1, length(basic)]) + 0.25 * std(Global);

mse = std(Global)

xlswrite('D:\reinforcement_learning\check_sensor\v4\data.xlsx', [Global', GPS', delta'])
