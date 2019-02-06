# cassie_rl_env
Some RL enviornments for training policies for the bipedal robot Cassie, following https://arxiv.org/abs/1803.05580. I try to follow the naming convention of OpenAI Gym, such as step, reset, but some details might be different.

I use a compiled version of the cassie simulator developed by Oregon State University Dynamic Robotics Laboratory, you can download the source code here: https://github.com/osudrl/cassie-mujoco-sim for possible extension.

Mujoco150 is required for simulation. You can download it from https://www.roboti.us/index.html and put it in the current folder.

cassieRlEnvMirror transform the state of the robot to its symmetric form every half step, enforcing symmetricity of the policy.

run mirror_test.py to simulate a pretrained policy.
