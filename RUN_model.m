clear;
%% Parameters for C23-L33-W20 terminal voltage for this motor is 12V
La= 0.94e-3; %H, inductance of the armature winding
Ra= 1; %Ohm, resistance of the armature winding
Kb=  0.0301; %V/rad/s the back emf constant 

N=10;  %the gear ratio
Kt=0.03; %Nm/Amp the torque constant

%% Parameters for robot body
M = 1000e-3; % Robot mass in KG
d = 0.01;%0.0; % Location of center of gravity of robot. x=d y=o in robot frame
R = 0.02;%radius of wheel in meter
L = 20e-2;%distance between wheel in meter
J = 0.01;% Kg*m^2 Moment inertia of disk robot KG*m^2 calculated using https://www.omnicalculator.com/physics/mass-moment-of-inertia

%% gain
k_eh = 1;           % S = e_h_dot + k_eh*e_h
k_ep = 1;
kseh = 40;          %S scaler
kseh_dot = 5e4;     %S dot scaler
ksep = 1;           %S scaler
ksep_dot = 100;     %S dot scaler

%% Input
tau_b = 1; %disturbance amplitude
f = 10;

G_x = -5; %When not using path generator
G_y = 0;  %When not using path generator

xa_init=0;
ya_init=0;
theta_init=deg2rad(0);

vol_gain = 1; % to slowdown the robot

TF = 6; %simulation time

%% Run simulation
path_generator;
simout = sim('model');%sim('copy_of_model2_disturbance');

% %% ----------Plot x vs y---------------
figure(4);
plot(simout.xa, simout.ya, G_x, G_y, '*', xa_init, ya_init, 'g*', 'LineWidth', 2, 'MarkerSize', 4), xlabel('x(m)'), ylabel('y(m)'), axis equal, grid on;
title(['robot path']);
legend({'robot path', 'goal position', 'start position'},'Location','northeast')
set(gca,'FontSize',12);
set(gca,'FontName','serif');
set(gca,'FontWeight','bold');
set(gca,'LineWidth',2);

%% -----------Plot actuator output-------------
figure(5);
%subplot 1
subplot(2,1,1),plot(simout.tout, simout.poser, 'LineWidth', 2), xlabel('t(s)'), ylabel('position error (m)'), grid on, hold on;
set(gca,'FontSize',12); %2
set(gca,'FontName','serif');
set(gca,'FontWeight','bold');
set(gca,'LineWidth',2);

%subplot 2
subplot(2,1,2),plot(simout.tout, simout.orrer, 'LineWidth', 2), xlabel('t(s)'), ylabel('heading error (deg)'), grid on, hold on;%, axis equal;
set(gca,'FontSize',12); %
set(gca,'FontName','serif');
set(gca,'FontWeight','bold');
set(gca,'LineWidth',2);

%% -----------Plot actuator output-------------
figure(6);
plot(simout.tout, simout.vl_flc, 'r', simout.tout, simout.vr_flc, 'g', 'LineWidth', 2), xlabel('t(s)'), ylabel('motor voltage (volt)'), grid on;%, axis equal;
legend({'left motor voltage', 'right motor voltage'},'Location','northeast')
set(gca,'FontSize',12); %
set(gca,'FontName','serif');
set(gca,'FontWeight','bold');
set(gca,'LineWidth',2);