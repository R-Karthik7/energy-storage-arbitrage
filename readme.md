# Energy Storage Arbitrage Using Dueling Deep Q-Network

A reinforcement learning project for optimizing the charging and discharging of an energy storage battery based on electricity market prices.

The project uses a **Dueling Deep Q-Network (Dueling DQN)** to learn battery charging, discharging, and idle decisions from historical electricity price data.

---

## Project Overview

Energy storage systems can store electricity when prices are low and discharge stored energy when prices are higher.

The objective of this project is to develop a reinforcement learning agent that learns an effective battery energy management policy from electricity price data.

The project includes:

* Battery energy storage simulation
* Electricity price-based environment
* Deep Q-Learning
* Dueling DQN architecture
* Experience replay
* Target network
* Epsilon-greedy exploration
* Real historical electricity price data
* Train/test data separation
* Random and price-based baseline agents
* Performance evaluation and visualization

---

## Objectives

The main objectives of the project are:

1. Model an energy storage system using a reinforcement learning environment.
2. Represent battery state using State of Charge (SOC), electricity price, and time information.
3. Train a Dueling DQN agent to select battery actions.
4. Evaluate the trained agent using unseen real electricity price data.
5. Compare the learned agent with baseline strategies.
6. Visualize the resulting performance and battery behavior.

---

## System Architecture

```text
Historical Electricity Price Data
              |
              v
       Data Preparation
              |
              v
     Train / Test Split
              |
              v
     Energy Storage Environment
              |
              v
        Battery Model
              |
              v
        Dueling DQN
              |
              v
      Action Selection
              |
       +------+------+
       |      |      |
       v      v      v
     Idle   Charge Discharge
              |
              v
       Battery State Update
              |
              v
          Reward
              |
              v
        Model Training
              |
              v
       Real Test Dataset
              |
              v
       Agent Evaluation
```

---

## Reinforcement Learning Environment

The environment represents the battery energy storage system.

### State

The agent receives three state variables:

```text
State = [SOC, Electricity Price, Hour]
```

Where:

* **SOC** = battery State of Charge
* **Electricity Price** = current electricity market price
* **Hour** = current time step within the day

The price and hour values are normalized before being provided to the neural network.

---

## Actions

The agent has three possible actions:

| Action | Description |
| ------ | ----------- |
| 0      | Idle        |
| 1      | Charge      |
| 2      | Discharge   |

### Idle

The battery does not charge or discharge.

### Charge

The battery purchases electricity and stores energy.

### Discharge

The battery releases stored energy and receives revenue based on the electricity price.

---

## Battery Model

The simulated battery has the following configuration:

| Parameter               |   Value |
| ----------------------- | ------: |
| Battery Capacity        | 100 kWh |
| Initial Energy          |  50 kWh |
| Minimum SOC             |     10% |
| Maximum SOC             |     90% |
| Maximum Charge Power    |   20 kW |
| Maximum Discharge Power |   20 kW |
| Charge Efficiency       |     90% |
| Discharge Efficiency    |     90% |

The battery constraints prevent the agent from charging beyond the maximum SOC or discharging below the minimum SOC.

---

## Reward Function

The reward is based on the economic effect of the selected battery action.

### Charging

Charging produces a negative reward because electricity is purchased.

```text
Charging Reward = - Energy Purchased × Electricity Price
```

### Discharging

Discharging produces a positive reward because stored electricity is sold.

```text
Discharging Reward = Energy Delivered × Electricity Price
```

### Idle

```text
Idle Reward = 0
```

A terminal battery value is also considered at the end of an episode based on the remaining battery energy and the final electricity price.

---

# Dueling DQN

The main reinforcement learning model is a **Dueling Deep Q-Network**.

Instead of directly producing Q-values through a single stream, the network separates the estimation into:

```text
                    Input State
                         |
                         v
                 Shared Network
                         |
              +----------+----------+
              |                     |
              v                     v
        Value Stream           Advantage Stream
             V(s)                  A(s,a)
              |                     |
              +----------+----------+
                         |
                         v
             Q(s,a) = V(s)
                    + A(s,a)
                    - mean(A(s,a))
                         |
                         v
                  Action Selection
```

The final Q-value is calculated as:

```text
Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
```

This allows the network to separately estimate the overall value of a state and the relative advantage of each action.

---

## DQN Components

The implementation includes:

* Dueling neural network
* Online Q-network
* Target network
* Experience replay buffer
* Epsilon-greedy exploration
* Discount factor
* Adam optimizer
* Periodic target network updates

The trained model is saved as:

```text
dueling_dqn_model.pth
```

---

# Dataset

The project uses real historical electricity price data.

The dataset was processed into:

```text
real_train_prices.csv
real_test_prices.csv
```

The test dataset contains:

```text
7,013 records
```

The observed test-data price range was:

```text
Minimum: €2.30/MWh
Maximum: €84.13/MWh
Average: €58.25/MWh
```

The data is converted into hourly price sequences for the energy storage environment.

---

# Training

The trained model is saved locally as:

dueling_dqn_model.pth

The model checkpoint is excluded from the GitHub repository through
.gitignore. To reproduce the project from a fresh clone, run:

python3 train_dueling_real.py

This creates the trained model locally. After training, the model can be
evaluated using:

python3 evaluate_dueling_real.py

---

# Evaluation

The trained model was evaluated using the separate real test dataset.

The evaluation was performed over 100 episodes with 24-hour episodes.

One evaluation run produced:

```text
Average Reward       : €2.94
Standard Deviation   : €0.45
Minimum Reward       : €0.66
Maximum Reward       : €4.11
Average Final SOC    : 33.18%

Charge Actions       : 200
Discharge Actions    : 800
Idle Actions         : 1400
```

The evaluation results are saved in:

```text
dueling_dqn_real_results.txt
```

---

# Baseline Comparison

The Dueling DQN was also compared with two baseline agents using the same real test dataset:

1. Random Agent
2. Price-Based Agent
3. Dueling DQN

The comparison experiment used 100 test episodes.

## Results

| Agent       | Average Reward | Std. Dev. | Average Final SOC |
| ----------- | -------------: | --------: | ----------------: |
| Random      |          €2.22 |     €0.71 |            44.92% |
| Price-Based |          €3.32 |     €0.67 |            46.04% |
| Dueling DQN |          €2.83 |     €0.59 |            35.32% |

### Action Distribution

| Agent       | Charge | Discharge | Idle |
| ----------- | -----: | --------: | ---: |
| Random      |    817 |       809 |  774 |
| Price-Based |    616 |       616 | 1168 |
| Dueling DQN |    198 |       822 | 1380 |

These results are the output of the final comparison experiment and are included as a reference for the current implementation.

The Dueling DQN achieved a higher average reward than the random baseline in this comparison, while the price-based benchmark achieved a higher average reward than the Dueling DQN.

---

# Generated Results

# Generated Results

The project can generate the following files locally:

```text
final_real_reward_comparison.png
final_real_soc_comparison.png
final_real_action_distribution.png

### Reward Comparison

`final_real_reward_comparison.png` compares the average reward achieved by the three agents.

### SOC Comparison

`final_real_soc_comparison.png` compares the average final battery State of Charge.

### Action Distribution

`final_real_action_distribution.png` shows the number of Charge, Discharge, and Idle actions selected by each agent.

---

# Project Structure

```text
energy-storage-arbitrage/
│
├── battery.py
├── energy_env.py
├── real_data.py
├── real_train_prices.csv
├── real_test_prices.csv
│
├── dueling_dqn.py
├── dueling_dqn_agent.py
├── replay_buffer.py
│
├── train_dueling_real.py
├── evaluate_dueling_real.py
├── diagnose_dueling_real.py
├── compare_real_agents.py
├── final_real_comparison.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

---

# Installation

Clone the repository:

```bash
git clone https://github.com/R-Karthik7/energy-storage-arbitrage.git
```

Move into the project directory:

```bash
cd energy-storage-arbitrage
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Train the Dueling DQN

```bash
python3 train_dueling_real.py
```

This trains the model using the real training dataset.

---

## 2. Evaluate the trained model

```bash
python3 evaluate_dueling_real.py
```

This evaluates the model using the real test dataset.

---

## 3. Diagnose the learned policy

```bash
python3 diagnose_dueling_real.py
```

This displays the model's action distribution and battery SOC behavior.

---

## 4. Compare agents

```bash
python3 compare_real_agents.py
```

This compares:

```text
Random Agent
Price-Based Agent
Dueling DQN
```

on the real test data.

---

## 5. Generate final graphs

```bash
python3 final_real_comparison.py
```

This generates the final performance, SOC, and action-distribution graphs.

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Gymnasium
* Reinforcement Learning
* Deep Q-Learning
* Dueling DQN

---

# Key Features

### Real Electricity Data

The project uses historical electricity price data instead of only manually generated prices.

### Battery Constraints

The environment models:

* Battery capacity
* SOC limits
* Charge/discharge power limits
* Charge efficiency
* Discharge efficiency

### Dueling DQN

The model separates state-value estimation and action-advantage estimation.

### Baseline Comparison

The learned policy is compared against random and price-based strategies.

### Train/Test Separation

Training and testing are performed using separate price datasets.

---

# Future Improvements

Possible future improvements include:

* Training on a larger historical dataset
* Using multiple electricity markets or pricing locations
* Adding renewable energy generation
* Adding battery degradation costs
* Including forecasted electricity prices
* Hyperparameter optimization
* Longer training runs
* Comparing Dueling DQN with Double DQN and other reinforcement learning algorithms
* Developing a web dashboard for battery operation visualization

---

# Disclaimer

This project is an experimental reinforcement learning simulation for energy storage arbitrage. The reported results depend on the selected dataset, battery parameters, reward formulation, training configuration, and evaluation methodology.

The project is intended for educational and experimental purposes.
