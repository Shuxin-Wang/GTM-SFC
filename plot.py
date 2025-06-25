import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def show_train_result(dir_path):
    agent_name_list = ['NCO', 'DRLSFCP', 'ActorEnhancedNCO', 'DDPG']
    agent_num = len(agent_name_list)

    df_list = []
    reward_list = []
    actor_loss_list = []
    critic_loss_list = []

    for agent_name in agent_name_list:
        csv_file_path = dir_path + '/' + agent_name + '.csv'
        df = pd.read_csv(csv_file_path)
        df_list.append(df)
        reward_list.append(df['Reward'])
        actor_loss_list.append(df['Actor Loss'])
        critic_loss_list.append(df['Critic Loss'])

    figure_path = 'save/result/plot/'
    colors = ['#925eb0', '#e29135', '#cc7c71', '#72b063', '#cc7c71']

    plt.figure()
    plt.title('Reward')
    for i in range(agent_num):
        plt.plot(reward_list[i], label=agent_name_list[i] + ' Reward', color=colors[i])
    plt.legend()
    plt.savefig(figure_path + 'Actor Loss.png', dpi=300)

    plt.figure()
    plt.title('Actor Loss')
    for i in range(agent_num):
        plt.plot(actor_loss_list[i], label=agent_name_list[i] + ' Actor Loss', color=colors[i])
    plt.legend()
    plt.savefig(figure_path + 'Actor Loss.png', dpi=300)

    plt.figure()
    plt.title('Critic Loss')
    for i in range(agent_num):
        plt.plot(critic_loss_list[i], label=agent_name_list[i] + ' Critic Loss', color=colors[i])
    plt.legend()
    plt.savefig(figure_path + 'Critic Loss.png', dpi=300)

    plt.show()

def show_evaluate_result(dir_path):
    agent_name_list = ['NCO', 'DRLSFCP', 'ActorEnhancedNCO', 'DDPG']

    # load csv files
    df_list = []
    for agent_name in agent_name_list:
        csv_file_path = dir_path + '/' + agent_name + '.csv'
        df = pd.read_csv(csv_file_path)
        df_list.append(df)

    # row number in csv files
    index_num = len(df_list[0]) if df_list else 0

    bar_width = 0.2
    index = np.arange(index_num)    # bar location
    labels = df_list[0]['Max SFC Length']
    colors = ['#925eb0', '#e29135', '#94c6cd', '#72b063', '#cc7c71']

    figure_path = 'save/result/plot/'

    for metric in df_list[0].columns.tolist()[1:]:
        plt.figure(figsize=(10, 6))
        for i, (df, agent_name) in enumerate(zip(df_list, agent_name_list)):
            plt.bar(
                index + i * bar_width,  # bar offset
                df[metric],
                width=bar_width,
                label=agent_name,
                color=colors[i]
            )

        # add data text
        for i, df in enumerate(df_list):
            for j, value in enumerate(df[metric]):
                plt.text(
                    j + i * bar_width,  # x
                    value + (0.01 if value >= 0 else -0.01) * max(abs(df[metric])),  # y
                    f'{value:.2f}',
                    ha='center',
                    va='bottom' if value >= 0 else 'top'
                )

        plt.xlabel('Max SFC Length', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(index + bar_width, labels=labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_path + metric + '.png', dpi=300)

    plt.show()

if __name__ == '__main__':
    show_train_result('save/result/train')
    # show_evaluate_result('save/result/evaluate')