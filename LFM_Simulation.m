%% 1.LFM信号分析
% （1）仿真LFM信号；
% （2）观察不同过采样率下的DFT结果；
% （3）观察不同TBP的LFM信号的频谱。
clc;clear;close all
%% （1）仿真LFM信号
%% 参数设置
B = 2e8;                        % 200MHz
T = 1e-6;                       % 脉冲持续时间
K = B/T;                        % 调频斜率    B=K*T
alpha_os = 1.25;                % 过采样率   alpha_os=fs/(K*T)
fs = alpha_os*B;                % 采样率
N = round( T / (1/fs) );        % 采样点数
dt = T/N;                       % 采样时间间隔
df = fs/N;                      % 采样频率间隔
t = -T/2:dt:T/2-dt;             % 时间变量
freq = -fs/2:df:fs/2-df;        % 频率变量
A_lfm = 1;                      % 设置信号幅度
f_lfm = 0;                      % 设置信号载频频率
y_lfm = A_lfm*exp(1j*(2*pi*f_lfm*t+pi*K*t.^2)); % 信号表达式
f = K*t;                        % 瞬时频率
phi = pi*K*t.^2;                % 瞬时相位

%% 绘制时域图
figure
subplot(121),plot(1e6*t,real(y_lfm));    % 绘制时域图
grid on;
xlabel('时间/ \mu s');           
ylabel('幅度');        
title( '线性调频信号时域波形');              
%% 绘制频谱图
Sf = fftshift(fft(fftshift(y_lfm)));  
subplot(122),plot(freq/1e6,abs(Sf) );    % 绘制频谱图
grid on;
xlabel('频率/MHz');          
ylabel('幅值');          
title('线性调频信号频谱');   

%% 绘制瞬时相位、瞬时频率图像
figure
subplot(121)
plot(t*1e+6,phi);grid on;
title('LFM信号相位');
xlabel('相对于t_0时间/\mus');
ylabel('相位/rad')
subplot(122)
plot(t*1e+6,f*1e-6);grid on;
title('LFM信号频率');
xlabel('相对于t_0时间/\mus');
ylabel('MHz');

%% 
TBP = 200;
T = 1e-6;                   % 脉冲持续时间
B = TBP/T;
K = B/T;                    % 调频斜率
alpha_os = 1.25;            % 过采样率
fs = alpha_os*B;            % 采样率
N = round( T / (1/fs) );    % 采样点数
dt = T/N;                   % 采样时间间隔
df = fs/N;                  % 采样频率间隔
t = -T/2:dt:T/2-dt;         % 时间变量
% t = 0:dt:T-dt;
freq = -fs/2:df:fs/2-df;    % 频率变量
A_lfm = 1;   % 设置信号幅度
f_lfm = 0;   % 设置信号载频频率
y_lfm = A_lfm*exp(1j*(2*pi*f_lfm*t+pi*K*t.^2)); % 信号表达式
Sf = fftshift(fft(fftshift(y_lfm)));    % Chirp信号频谱表达式
figure % 绘制基带LFM信号实部虚部波形
subplot(121),plot(t*1e+6,real(y_lfm)),grid on
title('信号实部'),xlabel('相对于t_0时间'),ylabel('幅度')
subplot(122),plot(t*1e+6,imag(y_lfm)),grid on
title('信号虚部'),xlabel('相对于t_0时间'),ylabel('幅度')
figure % 绘制基带LFM信号频域图
subplot(221),plot(freq*1e-6,real(Sf)),grid on
% axis([-40 40 -20 20])
title('a) 频谱实部'),ylabel('幅度')
subplot(222),plot(freq*1e-6,abs(Sf)),grid on
% axis([-20 20 0 40])
title('b) 频谱'),ylabel('幅度')
subplot(223),plot(freq*1e-6,imag(Sf)),grid on
% axis([-40 40 -20 20])
title('c) 频谱虚部'),xlabel('频率/MHz'),ylabel('幅度')
subplot(224),plot(freq*1e-6,unwrap(angle(Sf))),grid on
%axis([-50 50,0 900])
title('d) 频谱相位'),xlabel('频率/MHz'),ylabel('相位')
% title('线性调频信号的复频谱')

%% ======================================================================== %%
%% （2）观察不同过采样率下的DFT结果
alpha_0 = [0.8 1 1.2 1.4];% 过采样率
figure
for i = 1:length(alpha_0)
    TBP = 200;
    T = 1e-6;               % 脉冲持续时间
    B = TBP/T;
    K = B/T;                % 调频斜率
    fs = alpha_0(i)*B;      % 采样率
    N = round( T / (1/fs) );% 采样点数
    dt = T/N;               % 采样时间间隔
    df = fs/N;              % 采样频率间隔
    t = -T/2:dt:T/2-dt;     % 时间变量
    freq = -fs/2:df:fs/2-df;% 频率变量
    A_lfm = 1;              % 设置信号幅度
    f_lfm = 0;              % 设置信号载频频率
    y_lfm = A_lfm*exp(1j*(2*pi*f_lfm*t+pi*K*t.^2)); % 信号表达式
    Sf = fftshift(fft(fftshift(y_lfm)));    % Chirp信号频谱表达式
    %% 绘制频谱图
    subplot(length(alpha_0),1,i);
    plot(freq/1e6,abs(Sf) );grid on;
    axis([-200 200 0 inf]);
    if i==length(alpha_0)
        xlabel('频率/MHz');          
    end
    ylabel('幅值');          
    % title(['过采样率为',num2str(alpha_0(i))]); 
    % text(0,alpha_0(i),['过采样率为',num2str(alpha_0(i))],'HorizontalAlignment','center')
    legend(['过采样率为',num2str(alpha_0(i))])
end

% 参数设置
TBP = 200;             % 时间带宽积
T = 1e-6;             % 脉冲持续时间
alpha_0 = [1.6 1.4 1.2 1.0 0.8];             % 过采样率
figure; % title('过采样率\alpha_{os}在频谱中引起的能量间隙');
for i=1:length(alpha_0)
    % 参数计算
    B = TBP/T;             % 信号带宽
    K = B/T;               % 线性调频频率
    F = alpha_0(i)*B;      % 采样频率
    N = 2*ceil(F*T/2);     % 采样点数
    dt = T/N;              % 采样时间间隔
    df = F/N;              % 采样频率间隔
    % 变量设置
    t = -T/2:dt:T/2-dt;    % 时间变量
    f = -F/2:df:F/2-df;    % 频率变量
    f_zero = -F/2:F/(2*N):F/2-F/(2*N);    % 补零后的频率变零
    % 信号表达
    st = exp(1j*pi*K*t.^2);               % Chirp信号复数表达式
    Sf1 = fft(fftshift(st));              % Chirp信号频谱表达式
    st_zero = [st,zeros(1,N)];            % Chirp信号补零表达式
    Sf2 = fft(fftshift(st_zero));         % Chirp信号补零后的频谱表达式
    % 绘图
    subplot(length(alpha_0),2,2*i-1),plot(t*1e+6,real(st)),grid on
    if(i==1)
        title('信号实部')
    end
    if(i==length(alpha_0))
        xlabel('时间(\mus)')
    end
    subplot(length(alpha_0),2,2*i),plot(f_zero*1e-6,abs(Sf2)),grid on
    if(i==1)
        title('频谱幅度')
    end
    if(i==length(alpha_0))
        xlabel('频率单元/MHz')
    end
    % text(2.7,18,['\alpha_{os}= ',num2str(alpha_0(i))],'HorizontalAlignment','center')
    legend(['\alpha_{os}= ',num2str(alpha_0(i))])
end
%% ======================================================================== %%
%% （3）观察不同TBP的LFM信号的频谱
TBP = [25,50,100,400,1000];     % 时间带宽积
figure
for i = 1:length(TBP)
    T = 1e-6;                   % 脉冲持续时间
    B = TBP(i)/T;
    K = B/T;  % 调频斜率
    alpha_os = 2.5;% 过采样率
    fs = alpha_os*B; % 采样率
    N = round( T / (1/fs) );        % 采样点数
    t = linspace( -T/2 , T/2 , N);  % 
    y_lfm = A_lfm*exp(1j*(2*pi*f_lfm*t+pi*K*t.^2)); % 信号表达式

    %% 绘制频谱图
    freq = linspace(-fs/2,fs/2,N);  % 频域采样选取采样点，在-fs/2与fs/2间生成N个点
    Sf = fftshift( fft(y_lfm) );    
    subplot(length(TBP),1,i);
    plot(freq/1e6,abs(Sf) ),grid on;
    % axis([-200 200 0 1800]);
    if i==length(TBP)
        xlabel('频率/MHz');           
    end
    ylabel('幅度');  
    % title(['TBP为',num2str(TBP(i))]);
    legend(['TBP为',num2str(TBP(i))]);
end

TBP = [25,50,100,400,1000]; % 时间带宽积
figure
for i = 1:length(TBP)
    % 参数设置
    T = 1e-6;              % 脉冲持续时间
    B = TBP(i)/T;          % 信号带宽
    K = B/T;               % 线性调频频率
    alpha_os = 2.5;        % 过采样率
    F = alpha_os*B;        % 采样频率
    N = 2*ceil(F*T/2);     % 采样点数
    dt = T/N;              % 采样时间间隔
    df = F/N;              % 采样频率间隔
    % 变量设置
    t = -T/2:dt:T/2-dt;    % 时间变量
    freq = -F/2:df:F/2-df; % 频率变量
    % 信号表达
    y_lfm = A_lfm*exp(1j*(2*pi*f_lfm*t+pi*K*t.^2)); % 信号表达式
    Sf = fftshift(fft(fftshift(y_lfm)));     % Chirp信号频谱表达式
    %% 绘图
    % 频谱幅度
    subplot(5,2,2*i-1)
    plot(freq*1e-6,abs(Sf)),grid on
    if(i==5)
        xlabel('频率/MHz')
    end
    ylabel('幅度')
    line([-B*1e-6/2,-B*1e-6/2],[0,sqrt(1/K)*1e+6*N],'color','r','linestyle','--')
    line([ B*1e-6/2, B*1e-6/2],[0,sqrt(1/K)*1e+6*N],'color','r','linestyle','--')
    line([-B*1e-6/2, B*1e-6/2],[sqrt(1/K)*1e+6*N,sqrt(1/K)*1e+6*N],'color','r','linestyle','--')
    % 频谱相位
    subplot(5,2,2*i)
    plot(freq*1e-6,unwrap(angle(Sf))-max(unwrap(angle(Sf)))),hold on,grid on
    plot(freq*1e-6,(-pi*freq.^2/K)-max(-pi*freq.^2/K),'r--');
    set(gca,'YDir','reverse')    % 设置坐标轴翻转
    if(i==5)
        xlabel('频率/MHz')
    end
    ylabel('相位/rad')
    text(0,-TBP(i)/2,['TBP= ',num2str(TBP(i))],'HorizontalAlignment','center')
end

