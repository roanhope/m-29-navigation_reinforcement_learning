%% the lemniscate of Gerono, or lemniscate of Huygens, or figure-eight curve
t = 0:0.1: 2*pi; % t is between 1~2pi
s = 10;
x = cos(t)*s/2;
y = sin(2*t)/2*s;

% plot(x,y,'.');

robot_path = [x;y];