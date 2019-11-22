import tensorflow.keras as keras
import numpy as np
import xlrd
import collections
import random
import matplotlib.pyplot as plt

episodes = 300
e = 0.1

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
        self.model = self._build_model()
        self.gamma = 0.99

    def save_data(self, save_data_input):
        self._save_data.append(save_data_input)

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, activation="relu",
                                     input_dim=self.input_size))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.output_size, activation="linear"))
        model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
        return model

    def act(self, input):
        output = self.model.predict(input)
        if np.random.random() < e:
            for i in range(len(output)):
                output[i] += random.sample([-1, 1], 1)

        return output

    def update(self):
        mini_batch = random.sample(self._save_data, batch_size)
        temp = 2
        for state, k, count, loss, FAR, next_k, next_count, next_loss, next_FAR in mini_batch:
            if (next_loss - loss <= 0) & (next_FAR - FAR <= 0):
                k = k - 1 + temp * int(next_k - k > 0)
                count = count - 1 + temp * int(next_count - count > 0)
            elif (next_loss - loss > 0) & (next_FAR - FAR > 0):
                k = k + 1 - temp * int(next_k - k > 0)
                count = count + 1 - temp * int(next_count - count > 0)
            else:
                k = k - 1 + 1 * int(next_k - k > 0)
                count = count - 1 + 1 * int(next_count - count > 0)
            target = np.reshape([k, count], [1, 2])
            self.model.fit(state, target, epochs=1, verbose=0, batch_size=batch_size)

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
        j += 1
        # print(_count)
    next_loss = np.std(Global_data - height_corrected)
    next_FAR = FAR_count / len(GPS_data)
    return next_loss, next_FAR


if __name__ == "__main__":

    input_size = 3
    output_size = 2
    agent = Agent(input_size, output_size, len(GPS_data))
    k_save = []
    count_save = []
    loss_save = []
    FAR_save = []

    for i in range(episodes):

        k = random.randint(-5, 5)
        count = random.randint(0, 5)
        loss, FAR = cal_loss_and_FAR(GPS_data, Global_data, delta_data, k, count)

        for GPS, Global, delta in zip(GPS_data, Global_data, delta_data):
            data = [GPS, Global, delta]
            data = np.reshape(data, [1, input_size])
            # next_k, next_count = agent.act(data)  # output = [k, count]
            result = agent.act(data)  # output = [k, count]
            next_k = result[0, 0]
            next_count = result[0, 1]

            state = np.reshape([GPS, Global, delta], [1, input_size])
            next_loss, next_FAR = cal_loss_and_FAR(GPS_data, Global_data, delta_data, next_k, next_count)
            save_data_input = (state, k, count, loss, FAR, next_k, next_count, next_loss, next_FAR)
            agent.save_data(save_data_input)

            k = next_k
            count = next_count
            FAR = next_FAR
            loss = next_loss

            print("k = ", k, "count = ", count, "loss = ", loss, "FAR = ", FAR)
            k_save.append(k)
            count_save.append(count)
            loss_save.append(loss)
            FAR_save.append(FAR)

        agent.update()
        agent.reset()

    plt.plot(range(len(k_save)), k_save)
    plt.title("k")
    plt.show()

    plt.plot(range(len(count_save)), count_save)
    plt.title("count")
    plt.show()

    plt.plot(range(len(loss_save)), loss_save)
    plt.title("loss")
    plt.show()

    plt.plot(range(len(FAR_save)), FAR_save)
    plt.title("FAR")
    plt.show()
