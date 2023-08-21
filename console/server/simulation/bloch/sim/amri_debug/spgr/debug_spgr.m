clear; clc
%% 
figure(1); hold on
plot(m_store(1,1:20),'.-')
plot(m_store(2,1:20),'.-')
plot(m_store(3,1:20),'.-')
legend('Mx','My','Mz')

%% 
figure(2); hold on 
for u = 1:16
    plot(abs(signal(u,:)),'*')
end

%% 
load('stored_sim_magnetizations.mat'); 
figure(3); hold on 
imagesc(abs(fftshift(ifft2(signal))))
%% 
