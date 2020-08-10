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

G_x = -5; %When not using path generator
G_y = 0;  %When not using path generator

xa_init=0;
ya_init=0;
theta_init=deg2rad(0);

vol_gain = 11; % to slowdown the robot

% randomize G_x
in = setBlockParameter(in,blk,'Value',num2str(G_x));

% randomize G_y
in = setBlockParameter(in,blk,'Value',num2str(G_y));

%% Create Environment Interface
% Creating an environment model includes defining the following:
% 1. Action and observation signals that the agent uses to interact with the environment. For more information, see rlNumericSpec and rlFiniteSetSpec.
% 2. Reward signal that the agent uses to measure its success. For more information, see Define Reward Signals.
% Define the observation specification obsInfo and action specification actInfo.
obsInfo = rlNumericSpec([3 1],... %integral error, error, height
    'LowerLimit',[-inf -inf -inf]',...
    'UpperLimit',[inf inf inf]');
obsInfo.Name = 'observations';
obsInfo.Description = 'distance error and heading error fuzzified 22 classes each';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([2 1]);
actInfo.Name = 'volt R and volt L';
numActions = actInfo.Dimension(1);

%% Build the environment interface object.
env = rlSimulinkEnv('model','model/FLC/RL Agent',...
    obsInfo,actInfo);

load('trained_agent.mat','agent')

maxepisodes = 50000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',1000000);

simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,agent,simOpts);