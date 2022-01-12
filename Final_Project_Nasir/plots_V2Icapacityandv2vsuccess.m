
clc
clear all
N=1:1:10;
%marl =[38.89104 37.48961 36.40083 35.11014 34.02998 33.1631 32.30122 31.71619 30.81495 30.1115]
%marl_success =[0.9175 0.865 0.8475 0.8275 0.7925 0.7475 0.7025 0.6675 0.6425 0.6075]
sarl=[41.29147 40.67129 40.03836 39.40407 38.78399 38.15042 37.59182 37.04953 36.57266 36.09175];
sarl_success=[0.8975 0.8975 0.89 0.8875 0.88 0.87 0.8575 0.855 0.8475 0.845];
ran= [39.66697 37.45974 36.14133 35.09092 34.10049 33.19435 32.38625 31.66749 30.99287 30.42016 ];
ran_success= [0.9975 0.92 0.8675 0.85 0.8225 0.785 0.7475 0.72  0.6825 0.64];
hold on;grid on;box on;
 set(gcf,'color','w');
figure(1);
%plot(N,((smooth(smooth(marl)))),'r--o','LineWidth',1.2,'MarkerIndices',1:1:length(N))
plot(N,((smooth(smooth(sarl)))),'r--*','LineWidth',1.1,'MarkerIndices',1:1:length(N))
plot(N,((smooth(smooth(ran)))),'k--s','LineWidth',1.2,'MarkerIndices',1:1:length(N))
 %plot(N,((smooth(smooth(t_MPA) ))),'m--*','LineWidth',1.2,'MarkerIndices',1:1:length(N))
legend({'SARL','Random',},'Location','northeast')
xlabel('N x(8480) Bytes ')
ylabel('Average Sum V2I Capacity (Mbps) ')


figure(2);
hold on;grid on;box on;
set(gcf,'color','w');

%plot(N,((smooth(smooth(marl_success)))),'r--o','LineWidth',1.2,'MarkerIndices',1:1:length(N))
plot(N,((smooth(smooth(sarl_success)))),'r--*','LineWidth',1.1,'MarkerIndices',1:1:length(N))
plot(N,((smooth(smooth(ran_success)))),'k--s','LineWidth',1.2,'MarkerIndices',1:1:length(N) )
xlabel('N x (8480) Bytes ')
ylabel('Average V2V payload success probability')
legend({'SARL','Random',},'Location','northeast')
