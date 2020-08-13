clear;
%% load result
a = load('workspace');
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
tau_b = 0; %disturbance amplitude
f = 10;

G_x = 10; %When not using path generator
G_y = 0;  %When not using path generator

xa_init=0;
ya_init=0;
theta_init=deg2rad(0);

vol_gain = 10; % to slowdown the robot

Ts = 0.05;
Tf = 5;

% mode = 1; approaching point
% mode = 2; following point array
mode = 1; 

%% Create Environment Interface
% Creating an environment model includes defining the following:
% 1. Action and observation signals that the agent uses to interact with the environment. For more information, see rlNumericSpec and rlFiniteSetSpec.
% 2. Reward signal that the agent uses to measure its success. For more information, see Define Reward Signals.
% Define the observation specification obsInfo and action specification actInfo.
obsInfo = a.obsInfo;
obsInfo.Name = a.obsInfo.Name;
obsInfo.Description = a.obsInfo.Description;
numObservations = a.numObservations;

actInfo = a.actInfo;
actInfo.Name = a.actInfo.Name;
numActions = a.numActions;

%% Build the environment interface object.
env = a.env;

%% Fix the random generator seed for reproducibility.
% rng(0)

%% Create DDPG Agent
% Given observations and actions, a DDPG agent approximates the long-term
% reward using a critic value function representation. To create the
% critic, first create a deep neural network with two inputs, the
% observation and action, and one output. For more information on creating 
% a deep neural network value function representation, see "Create Policy 
% and Value Function Representations".

%     fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticStateFC2', ...
%             'Weights',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ... 
%             'Bias',2/sqrt(criticLayerSizes(1))*(rand(criticLayerSizes(2),1)-0.5))

%% CRITIC network
criticNetwork = a.criticNetwork;

%% Specify options for the critic representation using "rlRepresentationOptions".
criticOpts = a.criticOpts;

%% Create the critic representation using the specified deep neural network
% and options. You must also specify the action and observation specifications
% for the critic, which you obtain from the environment interface.
% For more information, see "rlQValueRepresentation".
% critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts); %only for matlab 2020a
critic = a.critic;

%% ACTOR network
% Given observations, a DDPG agent decides which action to take using an 
% actor representation. To create the actor, first create a deep neural
% network with one input, the observation, and one output, the action.

% Construct the actor in a similar manner to the critic.
% For more information, see "rlDeterministicActorRepresentation".

actorNetwork = a.actorNetwork;

actorOptions = a.actorOptions;

% actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions); %only for matlab 2020a
actor = a.actor;

%% AGENT
%To create the DDPG agent, first specify the DDPG agent options using rlDDPGAgentOptions.
agentOpts = a.agentOpts;
agentOpts.NoiseOptions.Variance = a.agentOpts.NoiseOptions.Variance;
agentOpts.NoiseOptions.VarianceDecayRate = a.agentOpts.NoiseOptions.VarianceDecayRate;

%% Then, create the DDPG agent using the specified actor representation, critic representation, and agent options. For more information, see rlDDPGAgent.
agent = a.agent;

%% Validate Trained Agent
% Validate the learned agent against the model by simulation.
path_generator;
set_param('model/G_x','Value',num2str(G_x))
set_param('model/G_y','Value',num2str(G_y))
simout = sim('model');

%% ----------Plot x vs y---------------
figure(4);
plot(simout.xa, simout.ya, simout.G_xt, simout.G_yt, '*', xa_init, ya_init, 'g*', 'LineWidth', 2, 'MarkerSize', 4), xlabel('x(m)'), ylabel('y(m)'), axis equal, grid on, hold on;
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
discrete_time = reshape(simout.vl_flc.time, 1, []);
vr = reshape(simout.vr_flc.signals.values, 1, []);
vl = reshape(simout.vl_flc.signals.values, 1, []);
plot(discrete_time, vr, 'r',discrete_time, vl, 'g', 'LineWidth', 2), xlabel('t(s)'), ylabel('motor voltage (volt)'), grid on;%, axis equal;
legend({'left motor voltage', 'right motor voltage'},'Location','northeast')
set(gca,'FontSize',12); %
set(gca,'FontName','serif');
set(gca,'FontWeight','bold');
set(gca,'LineWidth',2);