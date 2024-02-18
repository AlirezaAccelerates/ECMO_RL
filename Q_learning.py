class QLearningOffline:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, initial_q_value=0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_q_value = initial_q_value
        # Initialize Q-table with states as index and actions as columns, filled with initial_q_value
        self.q_table = pd.DataFrame(initial_q_value, index=np.arange(1,76), columns=np.arange(1,253))
        self.stepwise_q_values = pd.DataFrame(columns=['csn', 'state', 'action', 'q_value'])

    def update_q_table(self, state, action, reward, next_state, done, episode_id):
        max_future_q = 0 if next_state == 0 else self.q_table.loc[next_state].max()
        current_q = self.q_table.at[state, action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table.at[state, action] = new_q

        # Record the Q-value for this step
        new_row = pd.DataFrame([[episode_id, state, action, new_q]], columns=['csn', 'state', 'action', 'q_value'])
        self.stepwise_q_values = pd.concat([self.stepwise_q_values, new_row], ignore_index=True)

    def apply_final_reward_adjustment(self, episode_data, final_reward, episode_id):
        mortality_reward = -100 if final_reward > 0.5 else 100
        if isinstance(episode_data, np.ndarray):
            episode_data = pd.DataFrame(episode_data, columns=['state', 'action', 'reward_int', 'reward_fin', 'next_state', 'done'])
            
        # Adjust Q-values based on final reward
        for index, row in episode_data.iterrows():
            state, action = row['state'], row['action']
            adjusted_q_value = self.q_table.at[state, action] + self.learning_rate * mortality_reward
            self.q_table.at[state, action] = adjusted_q_value
            self.stepwise_q_values.loc[(self.stepwise_q_values['csn'] == episode_id) & (self.stepwise_q_values['state'] == state) & (self.stepwise_q_values['action'] == action), 'q_value'] = adjusted_q_value

    def save_q_table(self, filepath='./final_q_table_table.csv'):
        self.q_table.to_csv(filepath)

    def train(self, episodes_data, episode_ids):
        for episode_data, episode_id in zip(episodes_data, episode_ids):
            reward_buffer = 0
            if isinstance(episode_data, np.ndarray):
                episode_data = pd.DataFrame(episode_data, columns=['state', 'action', 'reward_int', 'reward_fin', 'next_state', 'done'])
            
            for index, row in episode_data.iterrows():
                state, action, reward_int, reward_fin, next_state, done = row[['state', 'action', 'reward_int', 'reward_fin', 'next_state', 'done']]
                reward_int_delta = reward_int - reward_buffer
                self.update_q_table(state, action, reward_int_delta, next_state, False, episode_id)
                reward_buffer = reward_int
            # Assuming last_state, last_action, etc. are defined correctly
            self.update_q_table(state, action, reward_fin, next_state, done, episode_id)
            self.apply_final_reward_adjustment(episode_data, reward_fin, episode_id)

    def get_stepwise_q_values(self):
        return self.stepwise_q_values

class EpisodeData:
    def __init__(self, dataframe, episode_id):
        self.dataframe = dataframe[dataframe['csn'] == episode_id]

    def preprocess(self):
        """Preprocess the data for a single episode, including handling intermediate and final rewards."""
        return self.dataframe[['cluster', 'action_number', 'SOFA_sigma', 'mortality_in_hospital', 'next_state', 'done']].values
      
data_train_int =data_train.copy()
data_train_int['cluster'] = data_train_int['cluster'].astype(int)
data_train_int['action_number'] = data_train_int['action_number'].astype(int)
data_train_int['next_state'] = data_train_int['next_state'].astype(int)
data_train_int['mortality_in_hospital'] = data_train_int['mortality_in_hospital'].astype(int)

unique_episodes = data_train_int['csn'].unique()
episodes_data = [(EpisodeData(data_train_int, ep).preprocess(), ep) for ep in unique_episodes]

q_learning_model = QLearningOffline()
q_learning_model.train([data for data, _ in episodes_data], [ep for _, ep in episodes_data])

# Save the final Q-table
q_learning_model.save_q_table('./final_q_table_table.csv')

# Get the DataFrame with stepwise Q-values and save to CSV
stepwise_q_values_df = q_learning_model.get_stepwise_q_values()
stepwise_q_values_df.to_csv('./stepwise_q_values_table.csv', index=False)
