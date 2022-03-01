%% match brainweb phantom to B0 map size (both to 32 x 32)
clear; clc 
load('brainweb.mat')
tm = brainweb.typemap; 
load('b0map_corr.mat')
%% 
figure(1); 
bwslice = tm(3:179,6:216,60);
b0slice = b0map(13:53,9:62,3);
subplot(121); imagesc(bwslice); colormap gray
subplot(122); imagesc(b0slice); colormap hot
%% 
[Xq,Yq] = meshgrid(linspace(1,54,32), linspace(1,41,32)); 
b0slice32 = interp2(b0slice,Xq,Yq,'Linear'); 
[Xp,Yp] = meshgrid(linspace(1,211,32), linspace(1,177,32)); 
bwslice32 = interp2(bwslice, Xp, Yp,'nearest'); 
%%
figure(2);

subplot(121); imagesc(bwslice32); axis off; title('Brainweb 32 x 32 typemap');
subplot(122); imagesc(b0slice32); axis off; title('B0 map [Hz]'); colorbar
colormap gray

%% 
b0units = 'Hz'; 

%% 
save('brainweb_sim_source_data.mat','bwslice32','b0slice32'); 