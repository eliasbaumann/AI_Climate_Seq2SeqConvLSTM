import tensorflow as tf
'''
Seq2Seq Convolutional LSTM
'''


class Seq2SeqEnc(tf.keras.Model):
    def __init__(self, n_filters, batch_size, dropout_rate):
        super(Seq2SeqEnc, self).__init__()
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.padding = tf.constant([[0,0],[0,0],[0,0],[1,1],[1,1]])
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.drop_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.drop_2 = tf.keras.layers.Dropout(self.dropout_rate)

        self.lstm_1 = tf.keras.layers.ConvLSTM2D(self.n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='sigmoid', return_sequences=True, return_state=True)
        self.lstm_2 = tf.keras.layers.ConvLSTM2D(self.n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='sigmoid', return_sequences=True, return_state=True)

    @tf.function
    def call(self, inputs, hidden, training=True, dropout=True):
        x,h_1,c_1 = self.lstm_1(self.pad(inputs), initial_state=hidden[0])
        x = self.drop_1(x, training=dropout)
        x = self.bn_1(x, training=training)
        x,h_2,c_2 = self.lstm_2(self.pad(x), initial_state=hidden[1])
        x = self.drop_2(x)
        return x, [[h_1, c_1],[h_2,c_2]]
    
    def initialize_hidden_state(self):
        return [[tf.zeros((self.batch_size, self.n_filters, 60,80)), tf.zeros((self.batch_size, self.n_filters,60,80))],[tf.zeros((self.batch_size, self.n_filters,60,80)), tf.zeros((self.batch_size, self.n_filters,60,80))]]

    def pad(self, inputs):
        return tf.pad(inputs, self.padding, "SYMMETRIC")


class Seq2SeqDec(tf.keras.Model):
    def __init__(self, n_filters, batch_size, dropout_rate, n_targets=3):
        super(Seq2SeqDec, self).__init__()
        self.n_filters = n_filters
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_targets = n_targets
        self.padding = tf.constant([[0,0],[0,0],[0,0],[1,1],[1,1]])
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.drop_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.drop_2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.drop_3 = tf.keras.layers.Dropout(self.dropout_rate)

        self.lstm_1 = tf.keras.layers.ConvLSTM2D(self.n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='sigmoid', return_sequences=True, return_state=True)
        self.lstm_2 = tf.keras.layers.ConvLSTM2D(self.n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='sigmoid', return_sequences=True, return_state=True)
        self.conv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(self.n_filters, kernel_size=(3,3), data_format='channels_first', activation='relu'))
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(self.n_targets, kernel_size=(3,3), data_format='channels_first', activation='relu'))

    @tf.function
    def call(self, inputs, hidden, training=True, dropout=True):
        x,h_3,c_3 = self.lstm_1(self.pad(inputs), initial_state=hidden[0])
        x = self.drop_1(x, training=dropout)
        x = self.bn_1(x, training=training)
        x,h_4,c_4 = self.lstm_2(self.pad(x), initial_state=hidden[1])
        x = self.drop_2(x, training=dropout)
        x = self.conv(self.pad(x))
        x = self.drop_3(x, training=dropout)
        x = self.out(self.pad(x))
        # x = self.drop_3(x)
        return x

    def pad(self, inputs):
        return tf.pad(inputs, self.padding, "SYMMETRIC")


# class Seq2SeqConvLSTM(tf.keras.Model):

#     def __init__(self, n_filters):
#         super(Seq2SeqConvLSTM, self).__init__()

#         self.padding = tf.constant([[0,0],[0,0],[0,0],[1,1],[1,1]])
#         self.bn_1 = tf.keras.layers.BatchNormalization()
#         self.bn_2 = tf.keras.layers.BatchNormalization()
#         self.encoder_1 = tf.keras.layers.ConvLSTM2D(n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='relu', return_sequences=True)
#         self.encoder_2 = tf.keras.layers.ConvLSTM2D(n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='relu', return_sequences=True)

#         self.decoder_1 = tf.keras.layers.ConvLSTM2D(n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='relu')
#         self.decoder_2 = tf.keras.layers.ConvLSTM2D(n_filters, kernel_size=[3,3], strides=(1,1), padding='valid', data_format='channels_first', activation='relu')

        
#         self.int_conv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=16, kernel_size=[3,3], strides=(1,1), activation='relu', data_format='channels_first'))

#         self.out_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=[3,3], strides=(1,1), activation='relu', data_format='channels_first')
#         self.out = tf.keras.layers.TimeDistributed(self.out_conv)


#     def call(self, inputs, training = True):
#         x = self.pad(inputs)
#         x = self.pad(self.int_conv(x))
#         x = self.bn_1(x, training=training)
#         # x = self.pad(self.encoder_1(x))
#         # x = self.pad(self.encoder_2(x))
#         out = self.out(x)
#         return out

#     def pad(self, inputs):
#         return tf.pad(inputs, self.padding, "SYMMETRIC")
