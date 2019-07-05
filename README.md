# cassie_rl_env
I use a compiled version of the cassie simulator, this is tested on ubuntu 16.04.

Mujoco150 is required for simulation. You can download it from https://www.roboti.us/index.html and put it in the current folder.

Some RL enviornments for training policies for the bipedal robot Cassie. I try to follow the naming convention of OpenAI Gym, such as step, reset, but some details might be different.

cassieRlEnvMirror transform the state of the robot to its symmetric form every half step, enforcing symmetricity of the policy.

Putorch is used to implement the neural network and learning algorithms.

run mirror_test.py to simulate a pretrained policy.

run RL_Mirror_Supervised.py to train policies. There are two methods in RL_Mirror_Supervised, train_policy_rl will train a policy from scratch, train_policy_rl_sl will train a policy by DASS samples from a pretrained policy. The default paramters can be used to train a stepping in place policy. To learn forward/backward walking, change the speed attribute in cassieRLEnvMirror.reset() accordingly.
