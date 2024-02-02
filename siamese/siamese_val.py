from keras.layers import Input, Conv2D, merge, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.regularizers import l2
from keras import backend as K
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
from LR_SGD import LR_SGD
from skimage.transform import warp, AffineTransform


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

    def __init__(self, Xtrain, Xval):
        self.Xval = Xval
        self.Xtrain = Xtrain
        self.n_classes, self.n_examples, self.w, self.h = Xtrain.shape
        self.n_val, self.n_ex_val, _, _ = Xval.shape

    def get_batch(self, n):
        """Create batch of n pairs, half same class, half different class"""
        m = n * 9
        lp = np.linspace(-0.00523599, 0.00523599, 11)
        lt = np.linspace(-1, 1, 11)
        lr = np.linspace(-0.04363325, 0.04363325, 11)
        categories = rng.choice(self.n_classes, size=(n,), replace=False)
        pairs = [np.zeros((n, self.h, self.w, 1)) for i in range(2)]
        affine_pairs = [np.zeros((m, self.h, self.w, 1)) for i in range(2)]
        targets = np.zeros((n,))
        affine_targets = np.zeros((m,))
        targets[n // 2:] = 1
        affine_targets[m // 2:] = 1
        count = 0
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0, self.n_examples)
            pairs[0][i, :, :, :] = self.Xtrain[category, idx_1].reshape(self.w, self.h, 1)
            idx_2 = rng.randint(0, self.n_examples)
            # pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n // 2 else (category +
                                                       rng.randint(1, self.n_classes)) % self.n_classes
            pairs[1][i, :, :, :] = self.Xtrain[category_2,
                                               idx_2].reshape(self.w, self.h, 1)
            affine_trans0 = pairs[0][i, :, :, :]
            affine_trans1 = pairs[1][i, :, :, :]
            for j in range(8):
                tform = AffineTransform(rotation=rng.choice(lr), translation=
                (rng.choice(lt), rng.choice(lt)), shear=rng.choice(lp))
                k = count + j
                img_warped0 = warp(affine_trans0, tform,
                                   output_shape=(img_rows, img_cols))
                affine_pairs[0][k, :, :, :] = img_warped0.reshape(self.w, self.h, 1)
                img_warped1 = warp(affine_trans1, tform,
                                   output_shape=(img_rows, img_cols))
                affine_pairs[1][k, :, :, :] = img_warped1.reshape(self.w, self.h, 1)
            count += 8
        return affine_pairs, affine_targets

    def make_oneshot_task(self, N):
        """Create pairs of test image, support set """
        # categories = rng.choice(self.n_val,size=(N,),replace=False)
        categories = rng.choice(self.n_val, size=(N,), replace=False)
        indices = rng.randint(0, self.n_ex_val, size=(N,))
        true_category = categories[0]
        ex1, ex2 = rng.choice(self.n_ex_val, replace=False, size=(2,))
        test_image = np.asarray([self.Xval[true_category,
                                 ex1, :, :]] * N).reshape(N, self.w, self.h, 1)
        support_set = self.Xval[categories, indices, :, :]
        support_set[0, :, :] = self.Xval[true_category, ex2]
        support_set = support_set.reshape(N, self.w, self.h, 1)
        pairs = [test_image, support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        return pairs, targets

    def test_oneshot(self, model, N, k, verbose=0):
        """Test average N way oneshot learning accuracy over k one-shot tasks"""
        pass
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks..."
                  .format(k, N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy"
                  .format(percent_correct, N))
        return percent_correct


# load training data
Xtrain = []
new_train = []
dirpath = os.getcwd()
data_path_train = dirpath + '/Train'
data_list_train = os.listdir(data_path_train)
# loops through each directory apending each image inside it
for dataset in data_list_train:
    #   print(dataset)
    img_list = os.listdir(data_path_train + '/' + dataset)
    #   print('Loaded the image of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path_train + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
        new_train.append(input_img_resize)
    Xtrain.append(new_train)
    new_train = []

Xtrain = np.array(Xtrain)
Xtrain = Xtrain.astype('float32')
print(Xtrain.shape)

# load validation data
Xval = []
new_val = []
data_path_val = dirpath + '/Validation'
data_list_val = os.listdir(data_path_val)
# loops through each directory apending each image inside it
for dataset in data_list_val:
    #   print(dataset)
    img_list = os.listdir(data_path_val + '/' + dataset)
    #   print('Loaded the image of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path_val + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
        new_val.append(input_img_resize)
    Xval.append(new_val)
    new_val = []
Xval = np.array(Xval)
Xval = Xval.astype('float32')
print(Xval.shape)

# Xtrain = np.ones((108,12,img_rows,img_cols))
# Xval = np.ones((33,10,img_rows,img_cols))

loader = Siamese_Loader(Xtrain, Xval)

evaluate_every = 80
loss_every = 20
batch_size = 16
N_way = 10
n_val = 60
# siamese_net.load_weights("my_model.h5")
best = 50
num_iterations = 100 * evaluate_every
val_acc_list = []
iterations_list = []
for i in range(num_iterations):
    (inputs, targets) = loader.get_batch(batch_size)
    loss = siamese_net.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        val_acc = loader.test_oneshot(siamese_net, N_way, n_val, verbose=True)
        val_acc_list.append(val_acc)
        iterations_list.append(i)
        if val_acc >= best:
            print("saving")
            siamese_net.save_weights("my_model6322.h5")
            best = val_acc
    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i, loss))

val_acc_list = np.array(val_acc_list)
iterations_list = np.array(iterations_list)

plt.figure(1, figsize=(7, 5))
plt.plot(iterations_list, val_acc_list)
plt.xlabel('num of iterations')
plt.ylabel('accuracy')
plt.title('Validation')
plt.grid(True)
plt.style.use(['classic'])
plt.savefig("Validation6322.png", bbox_inches='tight')
plt.show()



