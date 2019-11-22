import tensorflow.keras as keras
import numpy as np
import xlrd
import collections
import random
import matplotlib.pyplot as plt

episodes = 1000

x1 = xlrd.open_workbook("data.xlsx")
sheet1 = x1.sheet_by_index(0)
Global_data = sheet1.col_values(0)
GPS_data = sheet1.col_values(1)
delta_data = sheet1.col_values(2)

batch_size = int(0.7 * len(GPS_data))



class Agent:
    def __init__(self, input_size, output_size, length):
        self.input_size = input_size
        self.output_size = output_size
        self._save_data = collections.deque()
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()

    def save_data(self, save_data_input):
        self._save_data.append(save_data_input)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(200, activation="relu",
                                     input_dim=self.input_size))
        model.add(keras.layers.Dense(200, activation="relu"))
        model.add(keras.layers.Dense(self.output_size, activation="linear"))
        model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss="mse")
        return model

    def act(self, input):
        output = self.model.predict(input)
        k_output = output[0, 0:3]
        count_output = output[0, 3:6]
        k_action = np.argmax(k_output) - 1
        count_action = np.argmax(count_output) - 1
        if np.random.random() < self.epsilon:
            k_action = random.randint(-1, 1)
        if np.random.random() < self.epsilon:
            count_action = random.randint(-1, 1)

        return k_action, count_action

    def update(self, input, reward):

        # mini_batch, reward = self._save_data
        self.model.fit(input, reward, epochs=150, verbose=0, batch_size=batch_size)

        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        self._save_data = collections.deque()


# 计算误差值与虚警率（False Arm Rate, FAR）
def cal_loss_and_FAR(GPS_data, Global_data, delta_data, k, count):
    j = 0
    _count = 0
    height_corrected = np.zeros([len(GPS_data), 1])
    FAR_count = 0
    for _GPS, _Global, _delta in zip(GPS_data, Global_data, delta_data):
        if (_GPS < _Global + 3 * _delta) & (_GPS > _Global - 3 * _delta):
            height_corrected[j, 0] = _GPS
            _count = 0
        elif (_GPS > _Global + k * _delta) | (_GPS < _Global - k * _delta):
            height_corrected[j, 0] = _Global
            _count = 0
            if j < 0.5 * len(GPS_data):
                FAR_count += 1
        else:
            _count += 1
            if _count >= count:
                height_corrected[j, 0] = _Global
                if j < 0.5 * len(GPS_data):
                    FAR_count += 1
            else:
                height_corrected[j, 0] = _GPS

            # height_corrected[j, 0] = _GPS
        j += 1
        # print(_count)
    next_loss = np.std(Global_data - height_corrected)
    next_FAR = FAR_count / len(GPS_data)
    return next_loss, next_FAR


if __name__ == "__main__":

    input_size = int(0.7 * len(GPS_data))
    input_size = len(GPS_data)
    output_size = 6
    agent = Agent(input_size, output_size, len(GPS_data))
    k_save = []
    count_save = []
    loss_save = []
    FAR_save = []

    k = random.randint(0, 5)
    count = random.randint(0, 5)
    loss, FAR = cal_loss_and_FAR(GPS_data, Global_data, delta_data, k, count)

    # batch = np.append(GPS_data, Global_data, axis=0)
    # batch = np.append(batch, delta_data, axis=0)
    # batch = [GPS_data, Global_data, delta_data]
    # batch = np.reshape(batch, [len(GPS_data), 3])
    GPS_temp = np.reshape(GPS_data, [len(GPS_data), 1])
    Global_temp = np.reshape(Global_data, [len(Global_data), 1])
    delta_temp = np.reshape(delta_data, [len(delta_data), 1])
    batch = np.append(GPS_temp, Global_temp, axis=1)
    batch = np.append(batch, delta_temp, axis=1)

    for i in range(episodes):

        # batch = [GPS_data, Global_data, delta_data]
        # batch = np.reshape(batch, [len(GPS_data), 3])
        # row_rand = np.arange(batch.shape[0])
        # np.random.shuffle(row_rand)
        # mini_batch = batch[row_rand[0:input_size], :]
        # mini_batch = random.sample(batch, input_size)
        # row_rand = np.random.randint(0, len(GPS_data), input_size)
        mini_batch = batch
        input = mini_batch[:, 0] - mini_batch[:, 1]
        input = np.reshape(input, [1, len(input)])
        k_action, count_action = agent.act(input)
        reward = np.zeros([1, 6])

        # k和count一旦进入负值，很有可能就向负值方向不断发散，所以对k和count进行一定的限幅
        if k < 0:
            k = 3
        if count < 0:
            count = 1

        next_k = k + 0.5 * k_action
        next_count = count + 0.5 * count_action

        next_loss, next_FAR = cal_loss_and_FAR(mini_batch[:, 0], mini_batch[:, 1], mini_batch[:, 2], next_k, next_count)

        # if (next_loss <= loss) & (next_FAR <= FAR):
        #     reward[0, k_action + 1] = 2
        #     reward[0, count_action + 4] = 2
        # elif (next_loss > loss) & (next_FAR >= FAR):
        #     reward[0, k_action + 1] = -2
        #     reward[0, count_action + 4] = -2
        # else:
        #     reward[0, k_action + 1] = 0
        #     reward[0, count_action + 4] = 0
        # 加入各种限制条件，先多尝试一些条件
        if loss == 0:
            loss_ratio = 1
        else:
            loss_ratio = next_loss / loss

        if FAR == 0:
            FAR_ratio = 1
        else:
            FAR_ratio = next_FAR / FAR

        if (next_loss < loss) & (next_FAR < FAR):
            reward[0, k_action + 1] = 3
            reward[0, count_action + 4] = 3
        elif (next_loss == loss) & (next_FAR == FAR):
            reward[0, k_action + 1] = 0
            reward[0, count_action + 4] = 0
        elif loss_ratio * FAR_ratio < 1:
            reward[0, k_action + 1] = 1
            reward[0, count_action + 4] = 1
        else:
            reward[0, k_action + 1] = -3
            reward[0, count_action + 4] = -3

        # 抑制k和count相差过大，如果相差过大则奖励值减少
        if abs(k - count) > 10:
            reward[0, k_action + 1] -= 2
            reward[0, count_action + 4] -= 2

        data = (input, reward)
        agent.save_data(data)

        print(str(i) + "/" + str(episodes), "k = ", k, "count = ", count, "loss = ", loss, "FAR = ", FAR)

        k = next_k
        count = next_count
        loss = next_loss
        FAR = next_FAR

        k_save.append(k)
        count_save.append(count)
        loss_save.append(loss)
        FAR_save.append(FAR)

        agent.update(input, reward)
        agent.reset()

    plt.subplot(2, 2, 1)
    plt.plot(range(len(k_save)), k_save)
    plt.title("k")

    plt.subplot(2,2,2)
    plt.plot(range(len(count_save)), count_save)
    plt.title("count")

    plt.subplot(2, 2, 3)
    plt.plot(range(len(loss_save)), loss_save)
    plt.title("loss")

    plt.subplot(2, 2, 4)
    plt.plot(range(len(FAR_save)), FAR_save)
    plt.title("FAR")
    plt.show()
