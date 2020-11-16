import gym
from gym import wrappers

# ゲーム環境の構築
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "./movie", force=True)

# 1エピソードでの動き
for i_episode in range(1):

    # 環境の初期化
    observation = env.reset()

    for t in range(1000):

        # 行動の描画
        env.render()
        print(observation)

        # ランダムな行動を選択
        # 0 : カートを左へ
        # 1 : カートを右へ
        action = env.action_space.sample()

        # 行動の実行
        # observation : (カートの位置(-4.8 4.8)、カートの速度(-Inf Inf)、ポールの角度(-24 deg 24 deg)、ポールの角速度(-Inf Inf))
        # reward : 前の行動によって達成された報酬の量
        # done : 環境をリセットするべきかの判断(True : ゲームオーバー、False : コンティニュー)
        # info : デバッグに役立つ診断情報
        observation, reward, done, info = env.step(action)

        # ゲームオーバー後の処理
        if done == True:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()