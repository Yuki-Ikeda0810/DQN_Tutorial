import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


########################################################################
# 初期設定

# ゲーム環境の構築
env = gym.make('CartPole-v0').unwrapped

# matplotlibの初期設定
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# deviceの設定(GPUを使う場合)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################
# リプレイメモリ(遷移を保存する)

# ここでは、DQNを訓練するために、エクスペリエンス・リプレイ・メモリを定義している。
# これにより、エージェントが観測した遷移を記憶し、後でこのデータを再利用することができる。
# そして、ここからランダムにサンプリングすることで、バッチを構成する遷移は非相関化される。
# これにより、DQNの学習手順が大幅に安定化され、改善されることが示されている。

# Transitionは、環境内の単一の遷移を表す名前付きタプル
# (基本的には(state, action)のペアを、(next_state, reward)の結果にマッピングする。)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ReplayMemoryは、最近観測された遷移を保持するサイズに制限のあるサイクリックバッファ
# (学習のために遷移のランダムなバッチを選択する.sample() メソッドも実装されている。)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


########################################################################
# Qネットワークの定義

# ここでは、現在のスクリーンパッチと前のスクリーンパッチの差を取り込む畳み込みニューラルネットワークを定義している。
# これは、Q(s,left)とQ(s,right)を表す2つの出力を持っている(ここでsはネットワークへの入力です)。
# このネットワークは、現在の入力が与えられた各アクションを実行した場合の期待リターンを予測する。

class DQN(nn.Module):

    # DQNクラスのコンストラクタ
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)          # 2次元の畳み込み
        self.bn1 = nn.BatchNorm2d(16)                                   # 2次元のバッチ正規化
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)         # 2次元の畳み込み
        self.bn2 = nn.BatchNorm2d(32)                                   # 2次元のバッチ正規化
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)         # 2次元の畳み込み
        self.bn3 = nn.BatchNorm2d(32)                                   # 2次元のバッチ正規化

        
        # 入力画像のサイズを計算
        # (線形入力接続数は、conv2dレイヤーの出力に依存するため)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))    # 入力画像の幅を算出
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))    # 入力画像の高さを算出
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)               # 全結合(線型変換)

    # 順伝播
    # (次のアクションを決定するための1つの要素、)
    # (または最適化中のバッチのいずれかで呼び出される。)
    # (ensor([[left0exp,right0exp]...])を返す。)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))                             # 畳み込み -> バッチ正規化 -> 活性化(ReLU)
        x = F.relu(self.bn2(self.conv2(x)))                             # 畳み込み -> バッチ正規化 -> 活性化(ReLU)
        x = F.relu(self.bn3(self.conv3(x)))                             # 畳み込み -> バッチ正規化 -> 活性化(ReLU)
        return self.head(x.view(x.size(0), -1))                         # 行列の形状を変換 -> 全結合


########################################################################
# 入力抽出

# ここでは、環境からレンダリングされた画像を抽出して処理するためのユーティリティを定義している。
# torchvisionパッケージを用いて、画像変換を簡単に構成することができる。
# これらを実行すると、抽出したパッチの例が表示される。

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # Cartの中心

def get_screen():
    
    # トーチ順(CHW)に転置
    # (ジムから要求された返却画面は400x600x3であるが、800x1200x3などもっと大きい場合もあるため)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # 画面の上と下を切り取る(Cartが下半分にあるため)
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    # Cartの中央に正方形の枠が来るように、エッジを切り取る
    screen = screen[:, :, slice_range]

    # floatに変換、torch tensorに変換(これはコピーを必要としない)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # サイズ変更、バッチ寸法の追加(BCHW)
    return resize(screen).unsqueeze(0).to(device)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.show()


########################################################################
# トレーニング(ハイパーパラメータとユーティリティ)

# ここでは、モデルとそのオプティマイザのインスタンスを作成し、
# いくつかのユーティリティを定義している。

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# 画面サイズを取得し、形状に基づいて正しくレイヤーを初期化
# (gymから返ってきた、画面サイズは、この時点での典型的な寸法は3x40x90に近い。)
# (これは、get_screen()で取得し、縮小されたレンダーバッファの結果である。)
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# gymのアクションスペースからアクション数を取得
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

# select_actionは、epsilon greedy方の方策に従ってアクションを選択する。
# これは、アクションを選択するためにモデルを使うこともあれば、一様にサンプリングすることもある。
# ランダムなアクションを選択する確率は、EPS_STARTから始まり、EPS_ENDに向かって指数関数的に減衰していく。
# このとき、EPS_DECAY は減衰の速度を制御している。
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():

            # t.max(1)は、各行の最大カラムの値を返す
            # (max結果の2番目のカラムは、max要素が見つかった場所のインデックスなので、期待される報酬が大きいアクションを選ぶ。)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

# plot_durationsは、エピソードの持続時間をプロットするヘルパーで、
# 過去100エピソードの平均値(公式評価で使用されます)と一緒にプロットする。
# プロットは、メインのトレーニングループを含むセルの下に表示され、各エピソードの後に更新される。
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # 100のエピソードの平均を取って、それらをプロット
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 区画が更新されるように一時停止
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


########################################################################
# トレーニング(トレーニングループ：定義)

# ここでは、最適化の1つのステップを実行するoptimize_model関数を定義している。
# 最初にバッチをサンプリングし、すべての配列を1つに連結し、
# Q(st,at)とV(st+1)=maxaQ(st+1,a)を計算、それらを損失に結合する。
# 定義により、sが終端状態である場合、V(s)=0とする。
# また、V(st+1)を計算するためにターゲット・ネットワークを使用して安定性を高めていく。
# ターゲット・ネットワークの重みはほとんどの時間は凍結されているが、
# ポリシー・ネットワークの重みで更新される。
# これは通常は設定されたステップ数ですが、簡単にするためにエピソードを用いる。

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transitionsのバッチ配列を、バッチ配列のTransitionに変換
    batch = Transition(*zip(*transitions))

    # 非最終状態のマスクを計算し、バッチ要素を連結
    # (最終状態はシミュレーションが終了した後の状態)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a)を計算するには、モデルはQ(s_t)を計算し、次に、取られたアクションの列を選択
    # (これらは、policy_netに従って各バッチ状態に対して取られたであろうアクションである。)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # すべての次の状態についてV(s_{t+1})を計算
    # (non_final_next_statesのアクションの期待値は、"古い" target_netに基づいて計算される(max(1)[0]で最高の報酬を選択する)。)
    # (これはマスクに基づいて合わされ、状態が最終的なものであった場合には、期待される状態値か0のどちらかが得られるようになる。)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # 期待されるQ値を計算
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber損失を計算
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # モデルの最適化
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


########################################################################
# トレーニング(トレーニングループ：メイン)

# ここでは、メインのトレーニングループを示している。
# 最初に環境をリセットし、状態テンソルを初期化する。
# 次に、あるアクションをサンプリングし、それを実行し、
# 次の画面と報酬(常に1)を観察し、モデルを1回最適化する。
# エピソードが終了すると(モデルが失敗すると)、ループを再開する。

# num_episodesは小さく設定してあるので、
# 300以上のエピソードを実行してみるといい。

num_episodes = 50
for i_episode in range(num_episodes):

    # 環境と状態を初期化
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():

        # アクションを選択して実行
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 新しい状態を観察
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 遷移をメモリに保存
        memory.push(state, action, next_state, reward)

        # 次の状態に移動
        state = next_state

        # 最適化の1つのステップを実行(ターゲット・ネットワーク上)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    # DQN内のすべての重みとバイアスをコピーして、ターゲットネットワークを更新
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show() # plt.savefig('graph.png')