import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from collections import namedtuple
from myrtree import RTree
import argparse
random.seed(3)
special_dir = "model/NORMAL/"

DATASIZE_TYPE = {
    "TW1": 100000, "TW2": 200000, "TW5": 500000,
    "HW1": 1000000,
}

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
class PolicyNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)
    
    def forward(self, x):
        fc1_out = self.fc1(x)
        fc1_out_activation = F.relu(fc1_out)
        fc2_out = self.fc2(fc1_out_activation)
        out = F.softmax(fc2_out, dim=-1)
        return out

class ValueNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        fc1_out = self.fc1(x)
        fc1_out_activation = F.relu(fc1_out)
        fc2_out = self.fc2(fc1_out_activation)
        return fc2_out

class ACPPO1:
    def __init__(self, n_features, n_hidden, n_actions, lr_a, lr_c, lmbda, epochs, eps, gamma, device, buffer_size, epsilon):
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.value_net_output = 1

        self.lr_a = lr_a
        self.lr_c = lr_c

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

        self.buffer = []
        self.counter = 0
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.step = 0
        self.exploit_thresh = 0

        self.state_memory = []
        self.action_memory = []

    def build_net(self, optimizer_func):
        #  initialize instance of policy-net and value-net
        self.actor = PolicyNet(self.n_features, self.n_hidden, self.n_actions).to(self.device)
        self.critic = ValueNet(self.n_features, self.n_hidden, self.value_net_output).to(self.device)

        self.actor_optimizer = optimizer_func(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = optimizer_func(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, state, sp_flag=False):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(self.device)
        try:
            action_prob = self.actor(state)
            if torch.any(torch.isnan(action_prob)) or torch.any(action_prob <= 0):
                raise ValueError("Invalid action probabilities")
            
            dist = torch.distributions.Categorical(action_prob)
            if random.random() < 1 - self.epsilon:
                action = dist.probs.argmax().item()
            else:
                if sp_flag:
                    action = random.randint(0, 4)
                else:
                    action = random.randint(0, self.n_actions - 1)

            log_prob = dist.log_prob(torch.tensor(action, dtype=torch.int64).to(self.device))
            self.step += 1
        except (ValueError, RuntimeError) as e:
            print(f"Exception occurred in choose_action: {e}")
            print("EXCEPTION!!!!!!!")
            exit()
        return action, log_prob
    
    def store_transition(self, transiton):
        self.buffer.append(transiton)
        self.counter += 1

    def ppo_learn(self):
        states = torch.tensor(np.array([trans.state for trans in self.buffer]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([trans.action for trans in self.buffer]), dtype=torch.long).to(self.device).view(-1, 1)
        rewards = torch.tensor(np.array([trans.reward for trans in self.buffer]), dtype=torch.float32).to(self.device).view(-1, 1)
        next_states = torch.tensor(np.array([trans.next_state for trans in self.buffer]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([trans.dones for trans in self.buffer]), dtype=torch.float).to(self.device).view(-1, 1)

        next_q_target = self.critic(next_states).to(self.device)                       # target, next state_value
        td_target = rewards + self.gamma * next_q_target * (1 - dones)  # target, current state_value
        td_value = self.critic(states) .to(self.device)                                 # prediction, current state_value
        td_delta = td_target - td_value                                 # difference of current state-value between target and prediction

        # TD_errors
        td_delta = td_delta.cpu().detach().numpy()

        # calculate advantage function
        advantage = 0
        advantage_list = []

        for delta in td_delta[::-1]:
            # GAE func
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        advantage = torch.tensor(np.array(advantage_list), dtype=torch.float32).to(self.device)

        # .gather() alone axis 1, get all elems in actions, and calculate log-value of them
        # if policy doesn't change a lot, we can reuse this 'old_log_probs' which is in term of current episode
        # if policy changes a lot, we may recalculate this 'old_log_probs' from epochs loop
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # using the stored data to train model epochs times
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))        # new policy's action predictions
            ratio = torch.exp(log_probs - old_log_probs)                             # p(θ) / p'(θ)
            
            clip_item1 = ratio * advantage
            clip_item2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = -torch.mean(torch.min(clip_item1, clip_item2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # updating
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[:]

    def save_checkpoint(self, args):
        """
            args:
                training_args = dotdict({
                    'operation': "train"
                    'train_volume': train_volume,
                    'data_distribution': data_distribution,
                    'reference_tree_type': reference_tree_type,
                    'max_entry': max_entry,
                    'action_space_size': action_space_size
                })
        """
        filepath = special_dir + args.operation + "_" + args.train_volume + "_" + args.data_distribution + "_" + \
            args.reference_tree_type + "_" + str(args.max_entry) + "_" + str(args.action_space_size) + "_BestModel.pth"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, args):
        filepath = special_dir + args.operation + "_" + args.train_volume + "_" + args.data_distribution + "_" + \
            args.reference_tree_type + "_" + str(args.max_entry) + "_" + str(args.action_space_size) + "_BestModel.pth"
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(filepath, map_location=map_location)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print("Loaded checkpoint from {}".format(filepath))

def query_test(query_ratio, tree, query_type="RRQ"):
    x_min, x_max, y_min, y_max = 0, 100000, 0, 100000
    tree_acc_no = 0
    testing_query_area = query_ratio / 100 * ((x_max - x_min) * (y_max - y_min))    
    side = (testing_query_area**0.5) / 2
    k = 0
    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if query_type == "RRQ":
            # Execuate Random Range Query
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
                tree_access = tree.Query((x - side, y - side, x + side, y + side))
                tree_acc_no += tree_access
                k = k + 1
    if query_type == "RRQ":
        return [f"{query_ratio}% query", tree_acc_no / 1000], tree_acc_no / 1000
    
def query_test_ALL(query_ratio, tree, reference_tree, query_type="RRQ"):
    """
    Calculation of NAG need divide query node accesses of RTree to compare.
    Here are raw node access to show the improvement.
    """
    x_min, x_max, y_min, y_max = 0, 100000, 0, 100000
    tree_acc_no = 0
    ref_tree_acc_no = 0

    tree_acc_time = 0
    ref_tree_acc_time = 0
    testing_query_area = query_ratio / 100 * ((x_max - x_min) * (y_max - y_min))    
    side = (testing_query_area**0.5) / 2
    k = 0
    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if query_type == "RRQ":
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
                tree_acc_start_time = datetime.now()
                tree_access = tree.Query((x - side, y - side, x + side, y + side))
                tree_acc_time += (datetime.now() - tree_acc_start_time).total_seconds()

                ref_acc_start_time = datetime.now()
                reference_tree_access = reference_tree.Query((x - side, y - side, x + side, y + side))
                ref_tree_acc_time += (datetime.now() - ref_acc_start_time).total_seconds()

                tree_acc_no += tree_access
                ref_tree_acc_no += reference_tree_access
                k += 1
                print(f"query in ratio {query_ratio}%: {k}", end='\r')

    print(f"query time in ratio {query_ratio}: GSAR:{tree_acc_time}, rstar:{ref_tree_acc_time}")

    return {
        "query_ratio": query_ratio,
        "tree_acc": tree_acc_no / 1000,
        "ref_acc": ref_tree_acc_no / 1000
    }

import matplotlib.pyplot as plt
import numpy as np

def plot_query_results(test_result, title="Query Performance Comparison"):
    ratios = [res["query_ratio"] for res in test_result]
    tree_accs = [res["tree_acc"] for res in test_result]
    ref_accs = [res["ref_acc"] for res in test_result]

    x = np.arange(len(ratios))
    width = 0.4  
    plt.figure(figsize=(10,6))    
    plt.bar(x, ref_accs, width=width/2, label='R*-Tree', color='orange', alpha=0.7)
    plt.bar(x, tree_accs, width=width/2, label='GSAR', color='skyblue')

    plt.plot(x, ref_accs, 'o--', color='red', label='R*-Tree (line)')
    plt.plot(x, tree_accs, 'o--', color='blue', label='GSAR (line)')
    plt.xticks(x, ratios)
    plt.xlabel("Query ratio (%)")
    plt.ylabel("Average Node Access")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FEATURES_TYPE = {125: 4}
    np.random.seed(1)
    torch.manual_seed(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_or_test = {1: "training", 2:"testing"}
    parser = argparse.ArgumentParser(description="Select process mode for R-Tree experiment")
    parser.add_argument('--mode', type=int, choices=[1,2], default=2)
    parser.add_argument('--welltrained', type=bool, default=True)
    args = parser.parse_args()
    process_label = train_or_test[args.mode]
    directly_sp_args = False

    if args.welltrained:
        special_dir = "model/welltrained/"
        process_label = train_or_test[2]

    # data 
    data_distribution = "NORMAL"
    train_volume = "TW1"
    testing_volume = "TW2"
    file_format = ".npy"        # .npy .txt

    x_min, x_max, y_min, y_max = 0, 100000, 0, 100000
    data_edge_size = 1
    data_filename = "testing_data/" + train_volume + "_" + data_distribution + rf"{file_format}"
    testing_data_filename = "testing_data/testing_" + testing_volume + "_" + data_distribution + rf"{file_format}"
    training_set_size = DATASIZE_TYPE[train_volume]
    testing_set_size = DATASIZE_TYPE[testing_volume]

    # query
    query_reward_freq = 10
    training_query_area_ratio = 0.05

    # model save
    acppo2percent_q = 0
    min_acppo2percent_q = float('inf')
    load_bestModel_freq = 5

    # tree
    max_entry = 50
    min_entry_factor = 0.4
    
    feature_type = 125
    num_features = FEATURES_TYPE[feature_type]
    action_space_size = int(20)       
    state_space_size = action_space_size * (num_features)

    REFTREE_TYPE = {1: "rtree", 2: "rstar-tree"}
    reference_tree_type = REFTREE_TYPE[2]

    # pattern args  
    if max_entry == 50:
        training_args = dotdict({
            'operation': "train",
            'train_volume': train_volume,
            'data_distribution': data_distribution,
            'reference_tree_type': reference_tree_type,
            'max_entry': max_entry,
            'action_space_size': action_space_size
        })
    else:
        training_args = dotdict({
            'operation': f"3Layer-{max_entry}-train",
            'train_volume': train_volume,
            'data_distribution': data_distribution,
            'reference_tree_type': reference_tree_type,
            'max_entry': max_entry,
            'action_space_size': action_space_size
        })

    sp_args = dotdict({
        'operation': "self-play",
        'train_volume': train_volume,
        'data_distribution': data_distribution,
        'reference_tree_type': "self",
        'max_entry': max_entry,
        'action_space_size': action_space_size
    })
    dir_sp_args = dotdict({
        'operation': "dir-SP",
        'train_volume': train_volume,
        'data_distribution': data_distribution,
        'reference_tree_type': "self",
        'max_entry': max_entry,
        'action_space_size': action_space_size
    })
    
    testing_args = training_args

    # hyperparameters
    num_episodes = 20
    sp_num_episodes = 1
    gamma = 0.98
    lr_a = 1e-4
    lr_c = 1e-4
    n_hidden = 64
    lmbda=0.98
    epochs=10
    eps=0.2
    buffer_size = 20
    epsilon = 0.1

    # Load training data
    model_dataset = np.load(data_filename)
    testing_dataset = np.load(testing_data_filename)

    # Preparation work: set tree strategy
    acppo_tree = RTree(max_entry, min_entry_factor)
    acppo_tree.SetDefaultInsertStrategy("INS_AREA")
    acppo_tree.SetDefaultSplitStrategy("SPL_MIN_MARGIN")

    refer_tree = RTree(max_entry, min_entry_factor)
    if reference_tree_type == REFTREE_TYPE[1]:
        refer_tree.SetDefaultInsertStrategy("INS_AREA")
        refer_tree.SetDefaultSplitStrategy("SPL_MIN_AREA")
        
    elif reference_tree_type == REFTREE_TYPE[2]:
        refer_tree.SetDefaultInsertStrategy("INS_RSTAR")
        refer_tree.SetDefaultSplitStrategy("SPL_MIN_OVERLAP")

    agent = ACPPO1(
        n_features=state_space_size,
        n_hidden=n_hidden,
        n_actions=action_space_size,
        lr_a=lr_a,
        lr_c=lr_c,
        lmbda=lmbda,
        epochs=epochs,
        eps=eps,
        gamma=gamma,
        device=device,
        buffer_size=buffer_size,
        epsilon=epsilon
    )
    
    optim_func = torch.optim.Adam
    agent.build_net(optim_func)

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'dones'])
    
    queries = [2.0, 1.0, 0.5, 0.05]
    training_query_area = training_query_area_ratio / 100 * ( (x_max - x_min) * (y_max - y_min))      

    return_rewards_list = []
    test_result = []

    # First build a normal tree
    fixed_normal_rtree = RTree(max_entry, min_entry_factor)

    if reference_tree_type == REFTREE_TYPE[1]:
        # Guttman rtree
        fixed_normal_rtree.SetDefaultInsertStrategy("INS_AREA")
        fixed_normal_rtree.SetDefaultSplitStrategy("SPL_MIN_AREA")
    else:
        # rstar-tree
        fixed_normal_rtree.SetDefaultInsertStrategy("INS_RSTAR")
        fixed_normal_rtree.SetDefaultSplitStrategy("SPL_MIN_OVERLAP")

    for i in range(len(model_dataset)):
        insert_obj = model_dataset[i]
        if reference_tree_type == REFTREE_TYPE[1]:
            fixed_normal_rtree.DefaultInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])       
        elif reference_tree_type == REFTREE_TYPE[2]:
            fixed_normal_rtree.DirectInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])           
            fixed_normal_rtree.DirectSplitWithReinsert()

    # Model Training
    if process_label == "training":
        for i_episode in range(num_episodes):
            acppo_tree.Clear()
            refer_tree.Clear()
            
            rl_cnt = 0
            no_rl_cnt = 0
            episode_rewards = 0
            test_result = []

            for i in range(training_set_size):
                insert_obj = model_dataset[i]
                # temporarily traditional rtree
                if reference_tree_type == REFTREE_TYPE[1]:
                    refer_tree.DefaultInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                elif reference_tree_type == REFTREE_TYPE[2]:
                    refer_tree.DirectInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                    refer_tree.DirectSplitWithReinsert()
                acppo_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])

                while not acppo_tree.IsLeaf(acppo_tree.node_ptr):
                    if acppo_tree.GetMinAreaContainingChild() is None:
                        rl_cnt += 1
                        states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                        action, log_prob = agent.choose_action(states)
                        agent.state_memory.append(states)
                        agent.action_memory.append(action)
                        acppo_tree.InsertWithEvaluatedLoc(action)
                    else:
                        no_rl_cnt += 1
                        insert_loc = acppo_tree.GetMinAreaContainingChild()
                        acppo_tree.InsertWithLoc(insert_loc)
                acppo_tree.InsertWithLoc(0)

                if reference_tree_type == REFTREE_TYPE[1]:
                    acppo_tree.DefaultSplit()     # Guttman rtree
                elif reference_tree_type == REFTREE_TYPE[2]:
                    acppo_tree.DirectSplitWithReinsert()    # rstar rtree

                # leaves have no state to save
                if (len(agent.state_memory) % query_reward_freq == 0) and len(agent.state_memory) >= query_reward_freq:
                    avg_access_rate = 0
                    for k in range(query_reward_freq):
                        y_x_ratio = random.uniform(0.1, 1)
                        y_length = (training_query_area * y_x_ratio) ** 0.5
                        x_length = training_query_area / y_length

                        x_center = (model_dataset[i - k][0] + model_dataset[i - k][2]) / 2
                        y_center = (model_dataset[i - k][1] + model_dataset[i - k][3]) / 2
                        query_rec = [x_center - x_length / 2, y_center - y_length / 2, x_center + x_length / 2, y_center + y_length / 2]

                        refer_tree_query_rate = refer_tree.AccessRate(query_rec)
                        acppo_tree_query_rate = acppo_tree.AccessRate(query_rec)
                        avg_access_rate += (refer_tree_query_rate - acppo_tree_query_rate)
                    
                    idx = 0
                    records_num = len(agent.action_memory)
                    for idx in range(records_num):
                        _state = agent.state_memory[idx]
                        _action = agent.action_memory[idx]
                        _reward = avg_access_rate
                        if idx == records_num - 1:
                            _next_state = agent.state_memory[idx]
                            _done = True
                        else:
                            _next_state = agent.state_memory[idx + 1]
                            _done = False
                        trans = Transition(_state, _action, _reward, _next_state, _done)
                        agent.store_transition(trans)
                    
                    agent.ppo_learn()
                    agent.state_memory = []
                    agent.action_memory = []
                    episode_rewards += records_num * avg_access_rate
                    refer_tree.CopyTree(acppo_tree.tree_ptr)
                    print(f"Training Episode: {i_episode}\ttrained_data: {i}", end='\r')

            for qidx, ratio in enumerate(queries):
                if qidx == 0:
                    temp_res, acppo2percent_q = query_test(ratio, acppo_tree)
                else:
                    temp_res, _ = query_test(ratio, acppo_tree)
                test_result.append(temp_res)

            return_rewards_list.append(episode_rewards)
            print(f"qcompare: {test_result}")
            if acppo2percent_q < min_acppo2percent_q:
                min_acppo2percent_q = acppo2percent_q
                # save the best model
                print("save the best model", acppo2percent_q)
                agent.save_checkpoint(training_args)     
            if (i_episode + 1) % 5 == 0:
                agent.load_checkpoint(training_args)
        print("\n\nFinished Training\nCan start self-play or testing")
        process_label  = train_or_test[2]

    # Self-Play
    # process_label  = train_or_test[2]
    if process_label == "self-play":
        competitor = ACPPO1(
            n_features=state_space_size,
            n_hidden=n_hidden,
            n_actions=action_space_size,
            lr_a=lr_a,
            lr_c=lr_c,
            lmbda=lmbda,
            epochs=epochs,
            eps=eps,
            gamma=gamma,
            device=device,
            buffer_size=buffer_size,
            epsilon=epsilon
        )
        optim_func = torch.optim.Adam
        competitor.build_net(optim_func)

        if directly_sp_args is not True:
            # if it's continue self-play, execuate code below
            agent.load_checkpoint(training_args)
            agent.save_checkpoint(sp_args)
            competitor.load_checkpoint(sp_args)
        else:
            sp_args.operation = "dir-SP"

        for i_episode in range(sp_num_episodes):
            acppo_tree.Clear()
            refer_tree.Clear()

            episode_rewards = 0
            test_result = []

            for i in range(training_set_size):
                insert_obj = model_dataset[i]
                acppo_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                refer_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                ref_states = refer_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)

                while states is not None:
                    if acppo_tree.GetMinAreaContainingChild() is None:
                        action, log_prob = agent.choose_action(states)
                        agent.state_memory.append(states)
                        agent.action_memory.append(action)
                        acppo_tree.InsertWithEvaluatedLoc(action)
                    else:
                        insert_loc = acppo_tree.GetMinAreaContainingChild()
                        acppo_tree.InsertWithLoc(insert_loc)
                    states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                acppo_tree.InsertWithLoc(0)

                while ref_states is not None:
                    if refer_tree.GetMinAreaContainingChild() is None:
                        ref_action, _ = competitor.choose_action(ref_states)
                        competitor.state_memory.append(ref_states)
                        competitor.action_memory.append(ref_action)
                        refer_tree.InsertWithEvaluatedLoc(ref_action)
                    else:
                        ref_insert_loc = refer_tree.GetMinAreaContainingChild()
                        refer_tree.InsertWithLoc(ref_insert_loc)
                    ref_states = refer_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                refer_tree.InsertWithLoc(0)

                if reference_tree_type == REFTREE_TYPE[1]:
                    acppo_tree.DefaultSplit()     
                    refer_tree.DefaultSplit()
                elif reference_tree_type == REFTREE_TYPE[2]:
                    acppo_tree.DirectSplitWithReinsert()    
                    refer_tree.DirectSplitWithReinsert()
                
                if (len(agent.state_memory) % query_reward_freq == 0) and len(agent.state_memory) >= query_reward_freq:
                    avg_access_rate = 0
                    for k in range(query_reward_freq):
                        y_x_ratio = random.uniform(0.1, 1)
                        y_length = (training_query_area * y_x_ratio) ** 0.5
                        x_length = training_query_area / y_length

                        x_center = (model_dataset[i - k][0] + model_dataset[i - k][2]) / 2
                        y_center = (model_dataset[i - k][1] + model_dataset[i - k][3]) / 2
                        query_rec = [x_center - x_length / 2, y_center - y_length / 2, x_center + x_length / 2, y_center + y_length / 2]

                        refer_tree_query_rate = refer_tree.AccessRate(query_rec)
                        acppo_tree_query_rate = acppo_tree.AccessRate(query_rec)
                        avg_access_rate += (refer_tree_query_rate - acppo_tree_query_rate)
                    
                    idx = 0
                    records_num = len(agent.action_memory)
                    for idx in range(records_num):
                        _state = agent.state_memory[idx]
                        _action = agent.action_memory[idx]
                        _reward = avg_access_rate
                        if idx == records_num - 1:
                            _next_state = agent.state_memory[idx]
                            _done = True
                        else:
                            _next_state = agent.state_memory[idx + 1]
                            _done = False
                        trans = Transition(_state, _action, _reward, _next_state, _done)
                        agent.store_transition(trans)
                    
                    agent.ppo_learn()
                    agent.state_memory = []
                    agent.action_memory = []
                    episode_rewards += records_num * avg_access_rate
                    refer_tree.CopyTree(acppo_tree.tree_ptr)

                    print(f"Self-Play Episode: {i_episode}\ttrained_data: {i}", end='\r')

            for qidx, ratio in enumerate(queries):
                if qidx == 0:
                    temp_res, acppo2percent_q = query_test(ratio, acppo_tree)
                else:
                    temp_res, _ = query_test(ratio, acppo_tree)
                test_result.append(temp_res)

            print(f"qcompare: {test_result}")
            if acppo2percent_q < min_acppo2percent_q:
                min_acppo2percent_q = acppo2percent_q
                # save the best model
                print("save the best model", acppo2percent_q)
                agent.save_checkpoint(sp_args)  
                competitor.load_checkpoint(sp_args) 
            else:  
                if (i_episode + 1) % 5 == 0:
                    agent.load_checkpoint(sp_args)
        print("\n\nFinished Self-Playing\nCan start testing")

    # Testing
    if process_label == "testing":
        
        agent.load_checkpoint(testing_args)
        acppo_tree.Clear()
        refer_tree.Clear()
        rl_cnt = 0
        no_rl_cnt = 0
        test_result = []

        start_time = datetime.now()

        for i in range(testing_set_size):
            insert_obj = testing_dataset[i]

            refer_tree.DirectInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
            refer_tree.DirectSplitWithReinsert()

            acppo_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])           
            while not acppo_tree.IsLeaf(acppo_tree.node_ptr):
                if acppo_tree.GetMinAreaContainingChild() is None:
                    rl_cnt += 1
                    states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                    action, log_prob = agent.choose_action(states)
                    agent.state_memory.append(states)
                    agent.action_memory.append(action)
                    acppo_tree.InsertWithEvaluatedLoc(action)
                else:
                    no_rl_cnt += 1
                    insert_loc = acppo_tree.GetMinAreaContainingChild()
                    acppo_tree.InsertWithLoc(insert_loc)
            acppo_tree.InsertWithLoc(0)

            if reference_tree_type == REFTREE_TYPE[1]:
                acppo_tree.DefaultSplit()     # Guttman rtree
            elif reference_tree_type == REFTREE_TYPE[2]:
                acppo_tree.DirectSplitWithReinsert()    # rstar rtree

            print(f"Reconstructing the tree: inserting data {i}", end='\r')

        constructing_time = datetime.now() - start_time
        print("Reconsructing time is: ", constructing_time)
        print("Tree height: ", acppo_tree.GetTreeHeight())

        print("Query comparison")
        for qidx, ratio in enumerate(queries):
            temp_res = query_test_ALL(ratio, acppo_tree, refer_tree)
            test_result.append(temp_res)

        plot_query_results(test_result)

            
        print("\nconstruct tree and get final test result:")
        print(f"qcompare: {test_result}")
        
    print(f"==================================================\nFINISH\n==================================================")
