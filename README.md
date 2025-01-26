# TD3 Robot Learning

This repository contains a reinforcement learning (RL) project where a TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent is trained to control a robot (Panda arm) in a simulation environment to solve a task (e.g., opening a door).

The project uses the **Robosuite** library for simulation and **PyTorch** for deep learning, with an emphasis on stable policy improvement using the TD3 algorithm.

---

## Requirements

You can install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```
Make sure you have the necessary dependencies for Robosuite and PyTorch. You may need to install specific versions of these libraries based on your environment.

---

## Overview

This project leverages the TD3 algorithm to train a reinforcement learning agent. The environment used is a Panda robot arm in the Robosuite library, with the task being controlling the arm to open the door.

---

## Key Components:

`main.py`: Main entry point to train the TD3 agent. It initializes the environment, sets hyperparameters, and orchestrates the training process, saving the models periodically.

`td3_torch.py`: Contains the implementation of the TD3 agent, including the actor-critic networks, the replay buffer, and the learning process.

`buffer.py`: Implements a replay buffer that stores the experiences and samples them to train the agent.

`networks.py`: Defines the neural network architectures for the Actor and Critic models used in the TD3 algorithm.

`test.py`: A script to visualize the performance of the trained agent after training.

---

## Usage

## Training the Agent

To start training the agent, run the following command:
   ```
   python main.py
   ```
The agent will train for 1000 episodes by default, and periodically save the model checkpoints to the tmp/td3 directory. The training progress can be monitored via TensorBoard. After training, the agent's performance will be saved and can be loaded for further evaluation.

---

## Visualizing the Agent

After training the agent, you can visualize its performance using the test.py script:
   ```
   python test.py
   ```
This will load the saved model and run the agent in the environment, rendering the robot's actions.

---

## Hyperparameters

The following are the key hyperparameters used in the project:

- **actor_learning_rate**: Learning rate for the actor network (default: `1e-3`).
- **critic_learning_rate**: Learning rate for the critic network (default: `1e-3`).
- **tau**: Soft target network update parameter (default: `0.005`).
- **gamma**: Discount factor for future rewards (default: `0.99`).
- **update_actor_interval**: Number of steps before updating the actor network (default: `2`).
- **warmup**: Number of steps before training begins (default: `1000`).
- **n_actions**: Number of actions in the action space (default: `2`).
- **max_size**: Maximum size of the replay buffer (default: `1000000`).
- **layer1_size**: Size of the first hidden layer in the networks (default: `256`).
- **layer2_size**: Size of the second hidden layer in the networks (default: `128`).
- **batch_size**: Batch size used for training (default: `128`).
- **noise**: Noise added to the action for exploration (default: `0.1`).

These hyperparameters can be adjusted in the `main.py` file to suit different tasks or environments.

## Model Saving and Loading

The models (actor and critic networks) are saved periodically during training. The models can be loaded using the `load_models()` function in `td3_torch.py`:

   ```
   agent.load_models()
   ```
You can save the models at any point using:
   ```
   agent.save_models()
   ```

---

## License

This project is licensed under the MIT License.