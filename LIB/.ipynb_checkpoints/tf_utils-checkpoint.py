import tensorflow as tf


def lstm_layer(inputs, lengths, state_size, keep_prob=1.0, scope='lstm-layer', reuse=False, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        if return_final_state:
            return outputs, output_state
        else:
            return outputs


def temporal_convolution_layer(inputs, output_units, convolution_width, causal=False, dilation_rate=[1], bias=True,
                               activation=None, dropout=None, scope='temporal-convolution-layer', reuse=False):
    """
    Convolution over the temporal axis of sequence data.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps to use in convolution.
        causal: Output at timestep t is a function of inputs at or before timestep t.
        dilation_rate:  Dilation rate along temporal axis.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        if causal:
            shift = (convolution_width / 2) + (int(dilation_rate[0] - 1) / 2)
            pad = tf.zeros([tf.shape(inputs)[0], shift, inputs.shape.as_list()[2]])
            inputs = tf.concat([pad, inputs], axis=1)

        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[convolution_width, shape(inputs, 2), output_units]
        )

        z = tf.nn.convolution(inputs, W, padding='SAME', dilation_rate=dilation_rate)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        z = z[:, :-shift, :] if causal else z
        return z


def time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                                 dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None, dropout=None,
                scope='dense-layer', reuse=False):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].

    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def wavenet(x, dilations, filter_widths, skip_channels, residual_channels, scope='wavenet', reuse=False):
    """
    A stack of causal dilated convolutions with paramaterized residual and skip connections as described
    in the WaveNet paper (with some minor differences).

    Args:
        x: Input tensor of shape [batch size, max sequence length, input units].
        dilations: List of dilations for each layer.  len(dilations) is the number of layers
        filter_widths: List of filter widths.  Same length as dilations.
        skip_channels: Number of channels to use for skip connections.
        residual_channels: Number of channels to use for residual connections.

    Returns:
        Tensor of shape [batch size, max sequence length, len(dilations)*skip_channels].
    """
    with tf.variable_scope(scope, reuse=reuse):

        # wavenet uses 2x1 conv here
        inputs = time_distributed_dense_layer(x, residual_channels, activation=tf.nn.tanh, scope='x-proj')

        skip_outputs = []
        for i, (dilation, filter_width) in enumerate(zip(dilations, filter_widths)):
            dilated_conv = temporal_convolution_layer(
                inputs=inputs,
                output_units=2*residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='cnn-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            output_units = skip_channels + residual_channels
            outputs = time_distributed_dense_layer(dilated_conv, output_units, scope='cnn-{}-proj'.format(i))
            skips, residuals = tf.split(outputs, [skip_channels, residual_channels], axis=2)

            inputs += residuals
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        return skip_outputs


def sequence_log_loss(y, y_hat, sequence_lengths, max_sequence_length, eps=1e-15):
    """
    Calculates average log loss on variable length sequences.

    Args:
        y: Label tensor of shape [batch size, max_sequence_length, input units].
        y_hat: Prediction tensor, same shape as y.
        sequence_lengths: Sequence lengths.  Tensor of shape [batch_size].
        max_sequence_length: maximum length of padded sequence tensor.

    Returns:
        Log loss. 0-dimensional tensor.
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.minimum(tf.maximum(y_hat, eps), 1.0 - eps)
    log_losses = y*tf.log(y_hat) + (1.0 - y)*tf.log(1.0 - y_hat)
    sequence_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=max_sequence_length), tf.float32)
    avg_log_loss = -tf.reduce_sum(log_losses*sequence_mask) / tf.cast(tf.reduce_sum(sequence_lengths), tf.float32)
    return avg_log_loss


def sequence_rmse(y, y_hat, sequence_mask,sequence_lengths):
    """
    Calculates RMSE on variable length sequences.

    Args:
        y: Label tensor of shape [batch size, max_sequence_length, input units].
        y_hat: Prediction tensor, same shape as y.
        sequence_lengths: Sequence lengths.  Tensor of shape [batch_size].
        max_sequence_length: maximum length of padded sequence tensor.

    Returns:
        RMSE. 0-dimensional tensor.
    """
    y = tf.cast(y, tf.float32)
    squared_error = tf.square(y - y_hat)
    sequence_mask = tf.cast(sequence_mask, tf.float32)
    avg_squared_error = tf.reduce_sum(squared_error*sequence_mask) / tf.cast(tf.reduce_sum(sequence_lengths), tf.float32)
    rmse = tf.sqrt(avg_squared_error)
    return rmse


def log_loss(y, y_hat, eps=1e-15):
    """
    Calculates log loss between two tensors.

    Args:
        y: Label tensor.
        y_hat: Prediction tensor

    Returns:
        Log loss. 0-dimensional tensor.
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.minimum(tf.maximum(y_hat, eps), 1.0 - eps)
    log_loss = -tf.reduce_mean(y*tf.log(y_hat) + (1.0 - y)*tf.log(1.0 - y_hat))
    return log_loss


def rank(tensor):
    """Get tensor rank as python list"""
    return len(tensor.shape.as_list())


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]

    


def multi_layers_LSTM(inputs, lengths, state_size, keep_prob=1.0, scope='lstm-layer', reuse=False, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cells=[]
        for size in state_size :
            cells.append( tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    size,
                    reuse=reuse
                ),
                output_keep_prob=keep_prob
            ))

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=multi_rnn_cell,
            sequence_length=lengths,
            dtype=tf.float32
        )
        if return_final_state:
            return outputs, output_state
        else:
            return outputs



def Bidirectional_multi_layers_LSTM(inputs, lengths, state_size, keep_prob=1, scope='BiDir-lstm-layer', reuse=False, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cells=[]
        for size in state_size :
            cells.append( tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    size,
                    reuse=reuse,
                    state_is_tuple=True
                ),
                output_keep_prob=keep_prob,
                state_keep_prob=keep_prob
            ))

      
        outputs, output_state_fw, output_state_bw= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            inputs=inputs,
            cells_fw=cells,
            cells_bw=cells,
            sequence_length=lengths,
            dtype=tf.float32
        )
        if return_final_state:
            return outputs,output_state_fw, output_state_bw
        else:
            return outputs
def F1_score_TF(Y_true,Y_pred):
    with tf.variable_scope("F1_score_TF"):
        precision=tf.metrics.precision(Y_true,Y_pred)[0]
        recall=tf.metrics.recall(Y_true,Y_pred)[0]
        return 2 *tf.divide(precision*recall, precision+recall)
        
def  get_streaming_metrics(prediction,label,num_classes):

    with tf.name_scope("confusion_matrix"):
       
        batch_confusion = tf.confusion_matrix(label, prediction,
                                             num_classes=num_classes,
                                             name='batch_confusion')
        # Create an accumulator variable to hold the counts
        confusion = tf.Variable( tf.zeros([num_classes,num_classes],
                                          dtype=tf.int32 ),
                                 name='confusion' )
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign( confusion + batch_confusion )
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        confusion_image = tf.reshape( tf.cast( confusion, tf.float32),
                                  [1, num_classes, num_classes, 1])
   

        
    

    return confusion_update ,confusion_image




def output_layer(x, size=1, activation=None ,scope="output_layer"):
    return tf.contrib.layers.fully_connected(x, 
                                             size, 
                                             activation_fn=activation,
                                             scope=scope)



def  sigmoid_cross_entropy(y,y_hat)  :
     with tf.variable_scope("loss"):
   
        prediction=y_hat
        label=tf.cast(y,tf.float32)

        print("prediction",prediction)
        print("label",label)

        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=label))
        print("loss ",loss)
        return loss  
  
    
def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None, dropout=None,
                scope='dense-layer', reuse=False):

    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse,center=True, scale=True)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z
def rmse(prediction,target):
    return  tf.sqrt(tf.losses.mean_squared_error(prediction,target))
#################################################################################
def lstm_layer(inputs, lengths, state_size, keep_prob=1.0,
                   scope='lstm-layer',
                   reuse=False, return_final_state=False):

        with tf.variable_scope(scope, reuse=reuse):
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    state_size,
                    reuse=reuse
                ),
                output_keep_prob=keep_prob
            )
            outputs, output_state = tf.nn.dynamic_rnn(
                inputs=inputs,
                cell=cell_fw,
                sequence_length=lengths,
                dtype=tf.float32
            )
            if return_final_state:
                return outputs, output_state
            else:
                return outputs
def time_distributed_dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                         dropout=None, scope='time-distributed-dense-layer', reuse=False):

    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
    return z
#################################################################################
def embedding(self,Emb_dic,shape=[None]):

        
    for col  in  Emb_dic.keys():
        setattr(self,col,tf.placeholder(tf.int32,shape,name=col))       
        setattr(self,col+str("_embedding_var"),tf.get_variable(
                name=col+str("_embedding_var"),
                shape=Emb_dic[col],
                dtype=tf.float32
                ))

        setattr(self,col+str("_embedding"),
                tf.nn.embedding_lookup(getattr(self,col+str("_embedding_var"))
                                                              ,getattr(self,col),
                                                               col+str("_embedding")))

        print(col+str("_embedding"),Emb_dic[col],getattr(self,col+str("_embedding")))

    Embedding_list_tensors=[getattr(self,tensor+str("_embedding")) for  tensor in Emb_dic.keys() ]
    
    return tf.concat(Embedding_list_tensors,axis=1,name="Embedding_concat")
def place_holder(self,place_holders,shape=[None],T_type=tf.float32,axis=None):
    for col  in  place_holders:
        setattr(self,col,tf.placeholder(T_type,shape,name=col))  
    if len(place_holders)>1 :
        return tf.concat([tf.expand_dims(getattr(self,tensor ),axis=len(shape) ) for  tensor in place_holders  ],axis=len(shape))
    else :  
        return [getattr(self,tensor ) for  tensor in place_holders  ][0]
