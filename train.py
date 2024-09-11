import networkx as nx
import pandas as pd
import numpy as np
import stellargraph as sg
from imblearn import keras
from keras import regularizers, Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense, Multiply, Reshape, GlobalAveragePooling1D, Embedding, Conv1D, MaxPooling1D, \
    Flatten, Dropout, concatenate, Conv2D, Lambda
from keras.utils import to_categorical, pad_sequences
from sklearn.metrics._scorer import metric
from stellargraph.mapper import FullBatchNodeGenerator, GraphSAGENodeGenerator, RelationalFullBatchNodeGenerator
from stellargraph.layer import GCN, GCN_LSTM, GAT, GraphSAGE, RGCN
import tensorflow as tf
from sklearn import model_selection, metrics
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.regularizers import l2
from utils import MyLabelBinarizer, save_embeddings
from IPython.display import display, HTML
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow.keras.backend as K
# Set random seed
seed = 1701
np.random.seed(seed)
tf.random.set_seed(seed)

# 1.Data Preparation, Loading the Ant network
dataset = sg.datasets.xerces1_2()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

print(G.info())
node_subjects.value_counts().to_frame()
print(node_subjects.value_counts().to_frame())

# Splitting the data：8：1：1
train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.8, test_size=None, stratify=node_subjects, random_state=0
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=0.5, test_size=None, stratify=test_subjects, random_state=0
)
train_subjects.value_counts().to_frame()
print(train_subjects.value_counts().to_frame())
# Converting to numeric arrays
# LabelBinarizer cannot convert binary tags into one-hot encoded form,
# Encapsulate a Labelbinarizer function: MyLabelBinarizer
target_encoding = MyLabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)
# 2. Creating the GCN layers
# generator = RelationalFullBatchNodeGenerator(G)
# generator = Embed-SMOTE
generator = FullBatchNodeGenerator(G)
# train_gen = generator.flow(train_subjects.index, train_targets)
gcn = GCN(layer_sizes=[32, 32], activations=["relu", "relu"], generator=generator, dropout=0.1)
x_inp1, x_out1 = gcn.in_out_tensors()

gat = GCN(layer_sizes=[32, 32], activations=["relu", "relu"], generator=generator, dropout=0.1)
x_inp2, x_out2 = gat.in_out_tensors()
alpha = 0.5
concat = tf.keras.layers.Add()([(1 - alpha) * x_out1, alpha * x_out2])#使用add时，最后的维度还是不变，再进行pipeline时，就不会对y（bug）的长度产生影响
# dense1 = Dense(32, activation='relu')(concat)  # 1, none, 32
# x = tf.keras.layers.Dense(units=64, activation="relu")(dense1)  # 添加一个全连接层
# x = tf.keras.layers.Dropout(0.2)(x)  # 添加一个Dropout层，用于正则化
# x_all = tf.keras.layers.Dense(units=32, activation="relu")(x)  # 再次添加一个全连接层
predictions = Dense(2, activation='softmax', name="pred_Layer")(concat)  # 1, none, 2
# 3. Training and evaluating

merged_model = Model(inputs=[x_inp1, x_inp2], outputs=predictions)

merged_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=['categorical_crossentropy'],
                     metrics=['accuracy'])


train_gen1 = generator.flow(train_subjects.index, train_targets)
test_gen1 = generator.flow(test_subjects.index, test_targets)
train_gen = generator.flow(train_subjects.index, train_targets)
test_gen = generator.flow(test_subjects.index, test_targets)
val_gen = generator.flow(val_subjects.index,val_targets)
val_gen1 = generator.flow(val_subjects.index,val_targets)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=300,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)
history = merged_model.fit(x=[train_gen1.inputs, train_gen.inputs], y=train_gen.targets,
                           epochs=200, verbose=2,
                           validation_data=([test_gen1.inputs, test_gen.inputs], test_gen.targets),
                           shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
                           # callbacks=learning_rate_reduction
                           )

sg.utils.plot_history(history)
merged_model.evaluate(x=[test_gen1.inputs,test_gen.inputs], y=test_gen.targets, verbose=2)
test_predictions = merged_model.predict([test_gen1.inputs, test_gen.inputs])
test_predictions = test_predictions.squeeze(0)
test_pred = [np.argmax(one_hot) for one_hot in test_predictions]
labels = test_gen.targets.squeeze(0)
test_label = [np.argmax(one_hot) for one_hot in labels]

metric(test_label, test_pred)
# Making predictions with the model
all_nodes = node_subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = merged_model.predict([all_gen.inputs, all_gen.inputs])
# use the inverse_transform method of our target attribute specification to turn these values back to the original
# categories
node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
df.head(20)

# Node embeddings
embedding_model = tf.keras.Model(inputs=[x_inp1, x_inp2], outputs=concat)
emb = embedding_model.predict([all_gen.inputs, all_gen.inputs])
print(emb.shape)
X = emb.squeeze(0)
print(X.shape)
# save GCN embeddings
save_embeddings(X, "./data/xerces1_2/gcn_emb_dgcn.emd")
