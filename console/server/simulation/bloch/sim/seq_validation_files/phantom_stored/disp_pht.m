clear; clc 
g = load('grid.mat');
t1 = load('T1plane.mat');
t2 = load('T2plane.mat');
%% 
names = {'t1','t2','g'};
maps = {'T1map','T2map','PDmap'}; 
titles = {'T1map (s)','T2map (s)','PDmap'}; 

figure(99); 
for a = 1:3
    for b = 1:3
        subplot(3,3,3*(a-1)+b)
        pht = eval(names{a}); 
        imagesc(eval(sprintf('pht.%s', maps{b})));
        title(titles{b})
        axis equal off; colormap hot; colorbar
    end
end
