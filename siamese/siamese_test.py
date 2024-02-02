from keras.layers import Input, Conv2D, merge, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os, cv2
from LR_SGD import LR_SGD


def W_init(shape, name=None):
    """Initialize weights """
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias """
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def full_b_init(shape, name=None):
    """Initialize fully connected layer bias """
    values = rng.normal(loc=0, scale=2e-2, size=shape)
    return K.variable(values, name=name)


img_rows = img_cols = 128

input_shape = (img_rows, img_cols, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)


def Siamese_Network(siamese_input, a, b, c, d, e):
    x = Conv2D(32, (11, 11), activation='relu', input_shape=input_shape,
               kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
               name=a)(siamese_input)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (8, 8), activation='relu', kernel_regularizer=l2(2e-4),
               kernel_initializer=W_init, bias_initializer=b_init, name=b)(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (6, 6), activation='relu', kernel_initializer=W_init,
               kernel_regularizer=l2(2e-4), bias_initializer=b_init, name=c)(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (5, 5), activation='relu', kernel_initializer=W_init,
               kernel_regularizer=l2(2e-4), bias_initializer=b_init, name=d)(x)
    x = Flatten()(x)
    x = Dense(2048, activation="sigmoid", kernel_regularizer=l2(1e-3),
              kernel_initializer=W_init, bias_initializer=full_b_init, name=e)(x)
    return (x)


encoded_l = Siamese_Network(left_input, 'c11', 'c12', 'c13', 'c14', 'd11')
encoded_r = Siamese_Network(right_input, 'c21', 'c22', 'c23', 'c24', 'd21')
# merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0] - x[1])
# Eu_distance = lambda x: K.sqrt(K.square(x[0] - x[1]))
both = merge([encoded_l, encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1, activation='sigmoid', bias_initializer=full_b_init,
                   name='d3')(both)
siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

LR_mult_dict = {}
LR_mult_dict['c11'] = LR_mult_dict['c21'] = 1
LR_mult_dict['c12'] = LR_mult_dict['c22'] = 3
LR_mult_dict['c13'] = LR_mult_dict['c23'] = 5
LR_mult_dict['c14'] = LR_mult_dict['c24'] = 7
LR_mult_dict['d11'] = LR_mult_dict['d21'] = 9
LR_mult_dict['d3'] = 11

MN_mult_dict = {}
MN_mult_dict['c11'] = MN_mult_dict['c21'] = 0.85
MN_mult_dict['c12'] = MN_mult_dict['c22'] = 0.86
MN_mult_dict['c13'] = MN_mult_dict['c23'] = 0.87
MN_mult_dict['c14'] = MN_mult_dict['c24'] = 0.88
MN_mult_dict['d11'] = MN_mult_dict['d21'] = 0.89
MN_mult_dict['d3'] = 0.9

decay_rate = 0.01 / 80
optimizer = LR_SGD(lr=0.0009, decay=decay_rate, nesterov=True,
                   multipliers=LR_mult_dict, mn_multipliers=MN_mult_dict)
# optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
# optimizer = Adam(0.00006)
siamese_net.compile(loss=binary_crossentropy, optimizer=optimizer)

siamese_net.count_params()


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, Xtest):
        self.Xtest = Xtest
        _, self.n_ex_test, self.w, self.h = Xtest.shape

    def one_shot_testing(self, start, N, k):
        categories = np.arange(start, start + 10, 1)
        # indices = [0, 2, 3, 3, 1, 0, 2, 1, 3, 0]
        # indices = [0, 2, 3, 3, 1, 1, 2, 3, 2, 3]
        indices = rng.randint(0, self.n_ex_test, size=(N,))
        true_category = categories[k]
        ex1, ex2 = rng.choice(self.n_ex_test, replace=False, size=(2,))
        test_image = np.asarray([self.Xtest[true_category,
                                 ex1, :, :]] * N).reshape(N, self.w, self.h, 1)
        support_set = self.Xtest[categories, indices, :, :]
        support_set[k, :, :] = self.Xtest[true_category, ex2]
        support_set = support_set.reshape(N, self.w, self.h, 1)
        pairs = [test_image, support_set]
        targets = np.zeros((N,))
        targets[k] = 1
        return pairs, targets


# load test data
Xtest = []
new_test = []
dirpath = os.getcwd()
data_path_test = dirpath + '/Test'
data_list_test = os.listdir(data_path_test)
# loops through each directory apending each image inside it
for dataset in data_list_test:
    #   print(dataset)
    img_list = os.listdir(data_path_test + '/' + dataset)
    #   print('Loaded the image of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path_test + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
        new_test.append(input_img_resize)
    Xtest.append(new_test)
    new_test = []
Xtest = np.array(Xtest)
Xtest = Xtest.astype('float32')
print(Xtest.shape)

loader = Siamese_Loader(Xtest)

siamese_net.load_weights("my_model11855.h5")
N_test = 10
k_test = n_correct_test = index = 0

for i in range(3):
    for j in range(2):
        for k in range(N_test):
            inputs_test, targets_test = loader.one_shot_testing(index, N_test, k)
            probs_test = siamese_net.predict(inputs_test)
            k_test += 1
            if np.argmax(probs_test) == k:
                n_correct_test += 1
    index += 10

print("\nTesting model on {} unique {} way one-shot learning tasks ..."
      .format(k_test, N_test))
percent_correct_test = ((100 * n_correct_test) / k_test)
print("Got an average of {}% {} way one-shot testing accuracy"
      .format(percent_correct_test, N_test))
