clear; clc
%% TSE 
load('./simulated/tse/pe_info.mat');
load('./simulated/tse/tse32_signal_cylindrical.mat');
order_lin = order(:);
[A, order_inv] = sort(order_lin);
signal_rearr = signal(order_inv,:); 
image = abs(fftshift(ifft2(signal_rearr)));
figure(1);imagesc(image); title('TSE 32'); axis square; colormap gray;

%% IRSE 
load('./simulated/irse/irse32_signal_cylindrical.mat'); 
image = abs(fftshift(ifft2(signal)));
figure(2); imagesc(image); title('IRSE 32');axis square; colormap gray
load('./simulated/irse/irse32_signal_acr.mat');
image = abs(fftshift(ifft2(signal)));
figure(3); imagesc(image); title('IRSE 32 grid'); axis square; colormap gray
load('./simulated/irse/irse32_signal_acr2.mat');
image = abs(fftshift(ifft2(signal)));
figure(4); imagesc(image); title('IRSE 32 grid'); axis square; colormap gray
