%% 2.针对"基带LFM信号"实现脉冲压缩仿真
% （1）实现无误差的脉冲压缩，计算指标（IRW、PSLR、ISLR）
% （2）观察频域加窗的影响，计算指标（IRW、PSLR、ISLR）
clc;clear;close all
%% （1）针对"基带LFM信号",实现无误差的脉冲压缩，计算指标（IRW、PSLR、ISLR）
% 参数设置
TBP = [10 100];            % 时间带宽积
figure
for i=1:length(TBP)
    T = 10e-6;             % 脉冲持续时间
    % 参数计算
    B = TBP(i)/T;          % 信号带宽
    K = B/T;               % 线性调频频率
    alpha_os = 50;         % 过采样率，使用较高的过采样率是为了提高采样频率
    F = alpha_os*B;        % 采样频率
    N = 2*ceil(F*T/2);     % 采样点数
    dt = T/N;              % 采样时间间隔
    df = F/N;              % 采样频率间隔
    % 变量设置
    t = -T/2:dt:T/2-dt;    % 时间变量
    freq = -F/2:df:F/2-df;    % 频率变量
    t_out = linspace(2*t(1),2*t(end),2*length(t)-1);    % 循环卷积后的信号长度    
    % 信号表达
    st = exp(1j*pi*K*t.^2);               % Chirp信号复数表达式
    ht = conj(fliplr(st));                % 时域匹配滤波器
    sout = conv(st,ht);                   % 匹配滤波器输出
    sout = sout/max(sout);                % 归一化
    % 绘图
    subplot(1,length(TBP),i)
    plot(t_out*1e+6,real(sout)),grid on
    title(['TBP为',num2str(TBP(i))])
    axis([-4 4,-inf inf])
    xlabel('时间/\mus'),ylabel('幅度')
end


t_out = linspace(2*t(1),2*t(end),2*length(t)-1);    % 循环卷积后的信号长度    
% 信号表达
st = exp(1j*pi*K*t.^2);               % Chirp信号复数表达式
ht = conj(fliplr(st));                % 时域匹配滤波器
sout = conv(st,ht);                   % 匹配滤波器输出
% 信号变换
sout_nor = sout/max(sout);                          % 单位化
sout_log = 20*log10(abs(sout)./max(abs(sout))+eps); % 归一化
% 绘图
figure
subplot(221),plot(t*1e+6,real(st)),grid on
axis([-2 2,-inf inf])
title('(a)原始信号实部'),ylabel('幅度')

subplot(222),plot(t_out*1e+6,sout_log),grid on
axis([-0.5 0.5,-50 5])
title('(b)压缩后信号(经扩展)'),ylabel('幅度')
pslr = get_pslr(sout_log);
islr=get_islr(sout_nor);
hw = get_hw(sout_log);
hw = hw*dt;
% 压缩脉冲3dB宽度
% text(0,3,['PSLR= ',num2str(pslr),'dB'],'HorizontalAlignment','center')
text(0,3,['IRW= ',num2str(hw*1e+6),'\mus'],'HorizontalAlignment','center')

subplot(223),plot(t_out*1e+6,real(sout_nor)),grid on
axis([-2 2,-inf inf])
title('(c)压缩后信号')
xlabel('相对于t_0时间/\mus'),ylabel('幅度')
text(1,0.5,['PSLR= ',num2str(pslr),'dB'],'HorizontalAlignment','center')
text(1,0.6,['ISLR= ',num2str(real(islr)),'dB'],'HorizontalAlignment','center')

subplot(224),plot(t_out*1e+6,abs(angle(sout_nor))),grid on
axis([-0.5 0.5,-5 5])
title('(d)压缩后信号相位(经扩展)'),xlabel('相对于t_0时间/\mus'),ylabel('相位/rad')
% suptitle('基带线性调频信号的匹配滤波')

%% （2）观察频域加窗的影响，计算指标（IRW、PSLR、ISLR）
% 参数设置
TBP = 100;             % 时间带宽积
T = 10e-6;             % 脉冲持续时间
B = TBP/T;             % 信号带宽
K = B/T;               % 线性调频频率
alpha_os = 1.25;       % 过采样率
F = alpha_os*B;        % 采样频率
N = 2*ceil(F*T/2);     % 采样点数
dt = T/N;              % 采样时间间隔
df = F/N;              % 采样频率间隔
% 变量设置
t = -T/2:dt:T/2-dt;    % 时间变量
freq = -F/2:df:F/2-df; % 频率变量
% 信号表达
st = exp(1j*pi*K*t.^2);           % Chirp信号复数表达式
Sf = fft((st));                   % Chirp信号频谱表达式
Hf = exp(1j*pi*freq.^2/K);        % 频域匹配滤波器
% 窗函数
window = kaiser(N,2.5)';          % 时域窗  \beta 的一个典型值为2.5
Window = fftshift(window);        % 频域窗
% 信号变换
st_window = window.*exp(1j*pi*K*t.^2);          % 加窗后的Chirp信号
Hf_Window = Window.*Hf;                         % 加窗后的频域频谱滤波器
Soutf_Window = Hf_Window.*Sf;                   % 加窗后的匹配滤波器输出
% 绘图
figure
subplot(211),plot(freq*1e-6,Window)
axis([-5 5,0 1.2])
title('频域窗函数')
subplot(212),plot(freq*1e-6,real(Soutf_Window))
axis([-5 5,-15 15])
title('加窗后的频谱实部'),xlabel('频率/MHz')


% 参数设置
TBP = 100;             % 时间带宽积
T = 10e-6;             % 脉冲持续时间
B = TBP/T;             % 信号带宽
K = B/T;               % 线性调频频率
alpha_os = 100;        % 过采样率
F = alpha_os*B;        % 采样频率
N = 2*ceil(F*T/2);     % 采样点数
dt = T/N;              % 采样时间间隔
df = F/N;              % 采样频率间隔
% N_fft = 1024;
% 变量设置
t = -T/2:dt:T/2-dt;    % 时间变量
freq = -F/2:df:F/2-df; % 频率变量
% 信号表达
st = exp(1j*pi*K*t.^2);           % Chirp信号复数表达式
Sf = fft(st);                     % Chirp信号频谱表达式
h = zeros(1,N);
for i=1:N
    h(i)=conj(st(N-i+1));
end
Hf = fft(h);        % 频域匹配滤波器
S_out = abs(ifft(Sf.*Hf));
S_out_nor = S_out/max(S_out);
S_out_log = 20*log10(abs(S_out)./max(abs(S_out))+eps);

% 窗函数
window = kaiser(N,2.5)';          % 时域窗  \beta 的一个典型值为2.5
% window = hamming(N)';
h_window = conj(fliplr(st.*window));
L = 2*N-1;
Hf_Window = fft(h_window,L);
% s_W_out = abs(ifft(Sf.*Hf_Window));
s_W_out = ifft(fft(st,L).*Hf_Window);
sout_nor_W = s_W_out/max(s_W_out);                          % 单位化
sout_log_W = 20*log10(abs(s_W_out)./max(abs(s_W_out))+eps); % 归一化
% 绘图
% figure
% plot(Hf_Window);
% figure
% plot(abs(fftshift(Soutf_Window.*window)));%axis([-8 8 -inf inf])
% plot(freq/1e+6,fftshift(Hf_Window));%axis([-6 6 -inf inf])

pslr_W = get_pslr(sout_log_W);
islr_W = get_islr(sout_nor_W);
hw_W = get_hw(sout_log_W);
hw_W = hw_W*dt;

dtt = 2*T/L;
tt = -T:dtt:T-dtt;    % 时间变量
%% 绘图
figure
subplot(211),plot(t_out*1e+6,sout_log),grid on
axis([-1 1,-50 5])
ylabel('幅度')
% 压缩脉冲3dB宽度
text(0,3,['IRW= ',num2str(hw*1e+6),'\mus'],'HorizontalAlignment','center')
subplot(212),plot(tt*1e+6,sout_log_W),grid on
axis([-1 1,-50 5])
title('压缩后信号(经扩展)(加kaiser窗)')
xlabel('相对于t_0时间/\mus'),ylabel('幅度')
% 压缩脉冲3dB宽度
text(0,3,['IRW= ',num2str(hw_W*1e+6),'\mus'],'HorizontalAlignment','center')

figure
subplot(211),plot(t_out*1e+6,real(sout_nor)),grid on
axis([-1 1,-inf inf])
title('压缩后信号')
ylabel('幅度')
text(0.5,0.5,['PSLR= ',num2str(pslr),'dB'],'HorizontalAlignment','center')
text(0.5,0.6,['ISLR= ',num2str(real(islr)),'dB'],'HorizontalAlignment','center')
subplot(212),plot(tt*1e+6,real(sout_nor_W)),grid on
axis([-1 1,-inf inf])
title('压缩后信号(加kaiser窗)')
xlabel('相对于t_0时间/\mus'),ylabel('幅度')
text(0.5,0.5,['PSLR= ',num2str(pslr_W),'dB'],'HorizontalAlignment','center')
text(0.5,0.6,['ISLR= ',num2str(real(islr_W)),'dB'],'HorizontalAlignment','center')


% figure
% subplot(221),plot(t*1e+6,real(st)),grid on
% axis([-2 2,-inf inf])
% title('(a)原始信号实部'),ylabel('幅度')
% 
% subplot(222),plot(tt*1e+6,sout_log_W),grid on
% axis([-1 1,-50 5])
% title('(b)压缩后信号(经扩展)'),ylabel('幅度')
% % 压缩脉冲3dB宽度
% text(0,3,['IRW= ',num2str(hw_W*1e+6),'\mus'],'HorizontalAlignment','center')
% 
% subplot(223),plot(tt*1e+6,real(sout_nor_W)),grid on
% axis([-2 2,-inf inf])
% title('(c)压缩后信号')
% xlabel('相对于t_0时间/\mus'),ylabel('幅度')
% text(1,0.5,['PSLR= ',num2str(pslr_W),'dB'],'HorizontalAlignment','center')
% text(1,0.6,['ISLR= ',num2str(real(islr_W)),'dB'],'HorizontalAlignment','center')
% 
% subplot(224),plot(tt*1e+6,abs(angle(sout_nor_W))),grid on
% axis([-1 1,-5 5])
% title('(d)压缩后信号相位(经扩展)'),xlabel('相对于t_0时间/\mus'),ylabel('相位/rad')

%% 加Hamming窗
window_ham = hamming(N)';
h_window_ham = conj(fliplr(st.*window_ham));
L = 2*N-1;
Hf_Window_ham = fft(h_window_ham,L);
% s_W_out = abs(ifft(Sf.*Hf_Window));
s_W_out_ham = ifft(fft(st,L).*Hf_Window_ham);
sout_nor_W_ham = s_W_out_ham/max(s_W_out_ham);                          % 单位化
sout_log_W_ham = 20*log10(abs(s_W_out_ham)./max(abs(s_W_out_ham))+eps); % 归一化

%% 计算指标
pslr_W_ham = get_pslr(sout_log_W_ham);
islr_W_ham = get_islr(sout_nor_W_ham);
hw_W_ham = get_hw(sout_log_W_ham);
hw_W_ham = hw_W_ham*dt;

dtt = 2*T/L;
tt = -T:dtt:T-dtt;    % 时间变量
%% 绘图
figure
subplot(211),plot(t_out*1e+6,sout_log),grid on
title('压缩后信号'),axis([-1 1,-50 5])
ylabel('幅度')
% 压缩脉冲3dB宽度
text(0,3,['IRW= ',num2str(hw*1e+6),'\mus'],'HorizontalAlignment','center')
subplot(212),plot(tt*1e+6,sout_log_W_ham),grid on
axis([-1 1,-50 5])
title('压缩后信号(经扩展)(加hamming窗)')
xlabel('相对于t_0时间/\mus'),ylabel('幅度')
% 压缩脉冲3dB宽度
text(0,3,['IRW= ',num2str(hw_W_ham*1e+6),'\mus'],'HorizontalAlignment','center')

figure
subplot(211),plot(t_out*1e+6,real(sout_nor)),grid on
axis([-1 1,-0.5 1])
title('压缩后信号')
ylabel('幅度')
text(0.5,0.5,['PSLR= ',num2str(pslr),'dB'],'HorizontalAlignment','center')
text(0.5,0.6,['ISLR= ',num2str(real(islr)),'dB'],'HorizontalAlignment','center')
subplot(212),plot(tt*1e+6,real(sout_nor_W_ham)),grid on
axis([-1 1,-0.5 1])
title('压缩后信号(加hamming窗)')
xlabel('相对于t_0时间/\mus'),ylabel('幅度')
text(0.5,0.5,['PSLR= ',num2str(pslr_W_ham),'dB'],'HorizontalAlignment','center')
text(0.5,0.6,['ISLR= ',num2str(real(islr_W_ham)),'dB'],'HorizontalAlignment','center')

%% 加Hanning窗
window_han = hanning(N)';
h_window_han = conj(fliplr(st.*window_han));
L = 2*N-1;
Hf_Window_han = fft(h_window_han,L);
% s_W_out = abs(ifft(Sf.*Hf_Window));
s_W_out_han = ifft(fft(st,L).*Hf_Window_han);
sout_nor_W_han = s_W_out_han/max(s_W_out_han);                          % 单位化
sout_log_W_han = 20*log10(abs(s_W_out_han)./max(abs(s_W_out_han))+eps); % 归一化

%% 计算指标
pslr_W_han = get_pslr(sout_log_W_han);
islr_W_han = get_islr(sout_nor_W_han);
hw_W_han = get_hw(sout_log_W_han);
hw_W_han = hw_W_han*dt;

dtt = 2*T/L;
tt = -T:dtt:T-dtt;    % 时间变量
%% 绘图
figure
subplot(211),plot(t_out*1e+6,sout_log),grid on
title('压缩后信号'),axis([-1 1,-50 5])
ylabel('幅度')
% 压缩脉冲3dB宽度
text(0,3,['IRW= ',num2str(hw*1e+6),'\mus'],'HorizontalAlignment','center')
subplot(212),plot(tt*1e+6,sout_log_W_han),grid on
axis([-1 1,-50 5])
title('压缩后信号(经扩展、加hanning窗)')
xlabel('相对于t_0时间/\mus'),ylabel('幅度')
% 压缩脉冲3dB宽度
text(0,3,['IRW= ',num2str(hw_W_han*1e+6),'\mus'],'HorizontalAlignment','center')

figure
subplot(211),plot(t_out*1e+6,real(sout_nor)),grid on
axis([-1 1,-0.5 1])
title('压缩后信号'),ylabel('幅度')
text(0.5,0.5,['PSLR= ',num2str(pslr),'dB'],'HorizontalAlignment','center')
text(0.5,0.6,['ISLR= ',num2str(real(islr)),'dB'],'HorizontalAlignment','center')
subplot(212),plot(tt*1e+6,real(sout_nor_W_han)),grid on
axis([-1 1,-0.5 1])
title('压缩后信号(加hanning窗)')
xlabel('相对于t_0时间/\mus'),ylabel('幅度')
text(0.5,0.5,['PSLR= ',num2str(pslr_W_han),'dB'],'HorizontalAlignment','center')
text(0.5,0.6,['ISLR= ',num2str(real(islr_W_han)),'dB'],'HorizontalAlignment','center')


%% 函数实现代码
%% HW函数 IRW 冲激响应的3dB主瓣宽度
function [hw] = get_hw(Af)
    % 找到Af的最大位置
    [~,locmax] = max(Af);
    % 找到locmax左边最接近-3dB的位置
    [~,locleft] = min(abs(Af(1:locmax)+3));
    % 找到locmax右边最接近-3dB的位置
    [~,locright] = min(abs(Af(locmax:end)+3));
    locright = locright + locmax - 1;
    % 得到3dB波束宽度
    hw = locright-locleft;
end
%% PSLR函数 峰值旁瓣比，最大旁瓣与峰值的高度比
function [PSLR] = get_pslr(Af)
    % 找到所有的pesks
    peaks = findpeaks(Af);
    % 对peaks进行降序排列
    peaks = sort(peaks,'descend');
    % 得到第一旁瓣
    PSLR = peaks(2);
end

%
function islr=get_islr(x)

l=length(x);
a=find(x==max(x));
i=1;
for k=a-1:-1:2
    if(x(k)-x(k-1)<0&&x(k)-x(k+1)<0)
        lindian(i)=k;
        i=i+1;
    end
end
lindian1=max(lindian);

lindian=0;
i=1;
for k=a+1:l-1
    if(x(k)-x(k-1)<0&&x(k)-x(k+1)<0)
        lindian(i)=k;
        i=i+1;
    end
end
lindian2=min(lindian);

pmain=0;
for k=lindian1:lindian2
    pmain=pmain+x(k)^2;
end
x=x.^2;
ptotal=sum(x);
islr=10*log10((ptotal-pmain)/ptotal);
end
