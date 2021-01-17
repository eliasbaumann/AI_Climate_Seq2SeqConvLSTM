import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import h5py

from model import Seq2SeqEnc, Seq2SeqDec

'''
Theres a lot of code commented out which relates to a variation of the model which includes a second encoder with longer range lower resolution input.
'''


path = "INSERT"

BUFFER_SIZE = 100
BATCH_SIZE=2
SEQUENCE_LENGTH = 13
YEARS_PAST=4

ITERATIONS=10001

df = pd.read_hdf(path+"data.h5", 'df')
data = df.reset_index()[["t2m", "sd", "tp"]]
data["sd"] = np.log(data["sd"]+1)
data =  data.groupby(data.index // 2).mean()


elev = pickle.load(open(path+"elevation_dict_res_100.pkl", "rb"))
elev_eu = np.flip(np.asarray([[elev[x,y] for y in range(151,231)] for x in range(21,81)]), axis=0)
elev_eu = (elev_eu-elev_eu.min())/(elev_eu.max()-elev_eu.min())
elev_eu = np.tile(elev_eu[np.newaxis,:], (BATCH_SIZE, SEQUENCE_LENGTH, 3, 1, 1))

dec_in = tf.constant(elev_eu)

train_min = data.min()
train_max = data.max()
print(train_min)
print(train_max)

norm_data = (data-data.min())/(data.max()-data.min())
norm_data = norm_data.values.reshape(60,80,-1,3).transpose(2,3,0,1) # N,C,H,W
# data = data

train = norm_data
train_long = np.concatenate((train[:YEARS_PAST*52-SEQUENCE_LENGTH],train[:-SEQUENCE_LENGTH]),axis=0) # TODO assumption that a year has 52 weeks which is not quite correct
test = norm_data[norm_data.shape[0]-52:]
test_long = np.concatenate((test[:YEARS_PAST*52-SEQUENCE_LENGTH],test[:-SEQUENCE_LENGTH]),axis=0)



def get_xy_strides(data, data_long, seq_len):
    inp = data[:-seq_len]
    lab = data[seq_len:]
    res_inp = []
    res_lab = []
    res_inp_long = []
    for i in range(0,inp.shape[0]-seq_len+1):
        res_inp.append(inp[i:i+seq_len])
        res_lab.append(lab[i:i+seq_len])
        # res_lab.append(np.expand_dims(lab[i:i+seq_len,2,:,:], axis=1))
        tmp_long = data_long[i:i+(YEARS_PAST*52)]
        #tmp_long = np.reshape(tmp_long, (int(YEARS_PAST*52//SEQUENCE_LENGTH),-1)+tuple(tmp_long.shape[1:]))
        res_inp_long.append(np.mean(tmp_long, axis=0))
    return np.array(res_inp), np.array(res_lab), np.array(res_inp_long)

def ssd_loss(gt_labels, logits):
    return tf.reduce_sum(tf.square(gt_labels-logits), axis=None)



train_inp, train_lab, train_inp_long = get_xy_strides(train, train_long, seq_len=SEQUENCE_LENGTH)
test_inp, test_lab, test_inp_long = get_xy_strides(test, test_long, seq_len=SEQUENCE_LENGTH)

test_length = test_inp.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((train_inp,train_inp_long, train_lab)).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((test_inp,test_inp_long, test_lab)).batch(BATCH_SIZE, drop_remainder=True)


tf.keras.backend.clear_session()
optimizer = tf.keras.optimizers.Adam(lr=5e-4)
encoder = Seq2SeqEnc(52, BATCH_SIZE, 0.1)
# long_encoder = Seq2SeqEnc(32, BATCH_SIZE, .2)
decoder = Seq2SeqDec(52, BATCH_SIZE, 0.1)

# @tf.function
def train_step(x,x_l, y, hidden, hidden_l):
    with tf.GradientTape() as tape:
        _, enc_hid = encoder(x, hidden)
        # _, enc_hid_long = long_encoder(x_l, hidden_l)
        # enc_hidden_comb = [[(enc_hid[0][0]+enc_hid_long[0][0])/2.,(enc_hid[0][1]+enc_hid_long[0][1])/2.],[(enc_hid[1][0]+enc_hid_long[1][0])/2.,(enc_hid[1][1]+enc_hid_long[1][1])/2.]] #SUPER MEGA BAD SMELL :)
        #logit = decoder(dec_in, enc_hidden_comb)

        #dec_in = tf.zeros_like(x)
        
        logit = decoder(dec_in, enc_hid)
        loss = ssd_loss(y, logit)
    # variables = encoder.trainable_variables + long_encoder.trainable_variables + decoder.trainable_variables
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

def test_step(x,x_l, y, hidden, hidden_l):
    _, enc_hid = encoder(x, hidden)
    # _, enc_hid_long = long_encoder(x_l, hidden_l)
    # enc_hidden_comb = [[(enc_hid[0][0]+enc_hid_long[0][0])/2.,(enc_hid[0][1]+enc_hid_long[0][1])/2.],[(enc_hid[1][0]+enc_hid_long[1][0])/2.,(enc_hid[1][1]+enc_hid_long[1][1])/2.]] #SUPER MEGA BAD SMELL :)
    # logit = decoder(dec_in, enc_hidden_comb)
    #dec_in = tf.zeros_like(x)
    logit = decoder(dec_in, enc_hid, training=False, dropout=False)
    loss = ssd_loss(y, logit)
    return loss, logit

def predict(x, hidden):
    _, enc_hid = encoder(x, hidden)
    logit = decoder(dec_in, enc_hid, training=False, dropout=True)
    return logit

# steps per epoch: num_examples//batch_size

train_dataset = iter(train_dataset)
test_dataset = iter(test_dataset)

hidden_in = encoder.initialize_hidden_state()
hidden_in_long = 0 # long_encoder.initialize_hidden_state()

losses = []

for i in range(ITERATIONS):
    inp, inp_long, targ = next(train_dataset)
    batch_loss = train_step(inp,inp_long, targ, hidden_in, hidden_in_long)
    losses.append(batch_loss)
    if i %50 ==0:
        print("Batch:", i, "mean_loss_last_10:", np.mean(losses))
        losses = []

test_losses = []
preds = []

preds_future = []

print("trained.. starting test predictions")

for i, (inp, inp_long, targ) in enumerate(test_dataset):
    batch_loss, pred = test_step(inp,inp_long, targ, hidden_in, hidden_in_long)
    future = [predict(targ, hidden_in).numpy() for _ in range(50)]
    preds.append(pred.numpy())
    preds_future.append(future)
    test_losses.append(batch_loss.numpy())


hf = h5py.File(path+'predictions.h5', 'w')
hf.create_dataset('preds', data=np.array(preds))
hf.create_dataset('preds_future', data=np.array(preds_future))
hf.create_dataset('test_losses', data=np.array(test_losses))
hf.close()
