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

G_x = -5; %When not using path generator
G_y = 0;  %When not using path generator

xa_init=0;
ya_init=0;
theta_init=deg2rad(0);

vol_gain = 1; % to slowdown the robot

%% Create Environment Interface
% Creating an environment model includes defining the following:
% 1. Action and observation signals that the agent uses to interact with the environment. For more information, see rlNumericSpec and rlFiniteSetSpec.
% 2. Reward signal that the agent uses to measure its success. For more information, see Define Reward Signals.
% Define the observation specification obsInfo and action specification actInfo.
obsInfo = rlNumericSpec([2 1],... %integral error, error, height
    'LowerLimit',[-inf -inf]',...
    'UpperLimit',[ inf  inf]');
obsInfo.Name = 'observations';
obsInfo.Description = 'distance error and heading error';
numObservations = obsInfo.Dimension(1);

actInfo = rlNumericSpec([2 1]);
actInfo.Name = 'volt R and volt L';
numActions = actInfo.Dimension(1);

%% Build the environment interface object.
env = rlSimulinkEnv('model','model/FLC/RL Agent',...
    obsInfo,actInfo);


%% Set a custom reset function that randomizes the reference values for the model.
env.ResetFcn = @(in)localResetFcn(in);

%% Specify the simulation time Tf and the agent sample time Ts in seconds.
Ts = 1.0;
Tf = 60;

%% Fix the random generator seed for reproducibility.
rng(0)

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
criticLayerSizes = [50 50];
statePath = [
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
    fullyConnectedLayer(criticLayerSizes(1),'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(criticLayerSizes(2),'Name','CriticStateFC2')];
actionPath = [
    imageInputLayer([numActions 1 1],'Normalization','none','Name','Action')
    fullyConnectedLayer(criticLayerSizes(2),'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

%% View the critic network configuration.
% figure
% plot(criticNetwork)

%% Specify options for the critic representation using "rlRepresentationOptions".
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);

%% Create the critic representation using the specified deep neural network
% and options. You must also specify the action and observation specifications
% for the critic, which you obtain from the environment interface.
% For more information, see "rlQValueRepresentation".
% critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts); %only for matlab 2020a
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

%% ACTOR network
% Given observations, a DDPG agent decides which action to take using an 
% actor representation. To create the actor, first create a deep neural
% network with one input, the observation, and one output, the action.

% Construct the actor in a similar manner to the critic.
% For more information, see "rlDeterministicActorRepresentation".

actorNetwork = [
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name','actorFC')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

% actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions); %only for matlab 2020a
actor = rlRepresentation (actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);

%% AGENT
%To create the DDPG agent, first specify the DDPG agent options using rlDDPGAgentOptions.
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;

%% Then, create the DDPG agent using the specified actor representation, critic representation, and agent options. For more information, see rlDDPGAgent.
agent = rlDDPGAgent(actor,critic,agentOpts);

%% Train Agent
% To train the agent, first specify the training options. For this example, use the following options:
% 1. Run each training for at most 5000 episodes. Specify that each episode lasts for at most 200 time steps.
% 2. Display the training progress in the Episode Manager dialog box (set the Plots option) and disable the command line display (set the Verbose option to false).
% 3. Stop training when the agent receives an average cumulative reward greater than 800 over 20 consecutive episodes. At this point, the agent can control the level of water in the tank.
% For more information, see rlTrainingOptions.

maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',800);

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load('tflc_DDPG.mat','agent')
end

%% Validate Trained Agent
% Validate the learned agent against the model by simulation.

simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,agent,simOpts);

%% Local Function
function in = localResetFcn(in)
    % randomize G_x
    blk = sprintf('model/G_x');
    G_x = randn*5;
    in = setBlockParameter(in,blk,'Value',num2str(G_x));
    
    % randomize G_y
    blk = sprintf('model/G_y');
    G_y = randn*5;
    in = setBlockParameter(in,blk,'Value',num2str(G_y));
end