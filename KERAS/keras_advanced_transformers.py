# This builds an advanced transformer model for stock price prediction
# Install required libraries
# python3 -m pip install numpy pandas tensorflow pyarrow matplotlib requests scikit-learn
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import requests
from sklearn.preprocessing import MinMaxScaler 
from keras.layers import Layer, Dense, LayerNormalization, Dropout
import numpy as np
import pandas as pd

# Create a synthetic stock price dataset
np.random.seed(42)
data_length = 2000  # Adjust data length as needed
trend = np.linspace(100, 200, data_length)
noise = np.random.normal(0, 2, data_length)
synthetic_data = trend + noise
# Create a DataFrame and save as 'stock_prices.csv'
data = pd.DataFrame(synthetic_data, columns=['Close'])
data.to_csv('stock_prices.csv', index=False)
print("Synthetic stock_prices.csv created and loaded.")
# Load the dataset 
data = pd.read_csv('stock_prices.csv') 
data = data[['Close']].values 
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
# Prepare the data for training
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)
time_step = 100
X, Y = create_dataset(data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
print("Shape of X:", X.shape) 
print("Shape of Y:", Y.shape) 

'''
The MultiHeadSelfAttention layer implements the multi-head self-attention mechanism, which allows the model to focus on different parts of the input sequence simultaneously.
The attention parameter computes the attention scores and weighted sum of the values.
The split_heads parameter splits the input into multiple heads for parallel attention computation.
The call method applies the self-attention mechanism and combines the heads.
'''
# Implementing Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(Layer):  # Defines a custom Keras layer for multi-head self-attention.
    def __init__(self, embed_dim, num_heads=8):  # Initializes the layer with embedding dimension and number of heads.
        super(MultiHeadSelfAttention, self).__init__()  # Calls the parent class constructor.
        self.embed_dim = embed_dim  # Stores the embedding dimension.
        self.num_heads = num_heads  # Stores the number of attention heads.
        self.projection_dim = embed_dim // num_heads  # Calculates the dimension for each head.
        self.query_dense = Dense(embed_dim)  # Dense layer to project inputs to queries.
        self.key_dense = Dense(embed_dim)    # Dense layer to project inputs to keys.
        self.value_dense = Dense(embed_dim)  # Dense layer to project inputs to values.
        self.combine_heads = Dense(embed_dim)  # Dense layer to recombine all heads' outputs.

    def attention(self, query, key, value):  # Computes scaled dot-product attention.
        score = tf.matmul(query, key, transpose_b=True)  # Calculates attention scores by dot product of queries and keys.
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)  # Gets the dimension of the key vectors.
        scaled_score = score / tf.math.sqrt(dim_key)  # Scales scores to stabilize gradients.
        weights = tf.nn.softmax(scaled_score, axis=-1)  # Applies softmax to get attention weights.
        output = tf.matmul(weights, value)  # Computes weighted sum of values.
        return output, weights  # Returns the attention output and weights.

    def split_heads(self, x, batch_size):  # Splits the last dimension into (num_heads, projection_dim).
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))  # Reshapes input for multi-head attention.
        return tf.transpose(x, perm=[0, 2, 1, 3])  # Transposes to put heads before sequence dimension.

    def call(self, inputs):  # Defines the forward pass for the layer.
        batch_size = tf.shape(inputs)[0]  # Gets the batch size.
        query = self.query_dense(inputs)  # Projects inputs to queries.
        key = self.key_dense(inputs)      # Projects inputs to keys.
        value = self.value_dense(inputs)  # Projects inputs to values.
        query = self.split_heads(query, batch_size)  # Splits queries into multiple heads.
        key = self.split_heads(key, batch_size)      # Splits keys into multiple heads.
        value = self.split_heads(value, batch_size)  # Splits values into multiple heads.
        attention, _ = self.attention(query, key, value)  # Computes attention output.
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # Transposes back to original format.
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # Concatenates all heads.
        output = self.combine_heads(concat_attention)  # Projects concatenated output back to embedding dimension.
        return output  # Returns the final output of the multi-head attention layer.

#Implementing Transformer Block

class TransformerBlock(Layer):  # Defines a custom Keras layer for a transformer block.
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):  # Initializes the transformer block.
        super(TransformerBlock, self).__init__()  # Calls the parent class constructor.
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)  # Multi-head self-attention layer.
        self.ffn = tf.keras.Sequential([  # Feed-forward neural network.
            Dense(ff_dim, activation="relu"),  # First dense layer with ReLU activation.
            Dense(embed_dim),  # Second dense layer projecting back to embedding dimension.
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # Layer normalization after attention.
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # Layer normalization after feed-forward.
        self.dropout1 = Dropout(rate)  # Dropout after attention.
        self.dropout2 = Dropout(rate)  # Dropout after feed-forward.

    def call(self, inputs, training):  # Defines the forward pass for the transformer block.
        attn_output = self.att(inputs)  # Applies multi-head self-attention.
        attn_output = self.dropout1(attn_output, training=training)  # Applies dropout to attention output.
        out1 = self.layernorm1(inputs + attn_output)  # Adds residual connection and normalizes.
        ffn_output = self.ffn(out1)  # Applies feed-forward network.
        ffn_output = self.dropout2(ffn_output, training=training)  # Applies dropout to feed-forward output.
        return self.layernorm2(out1 + ffn_output)  # Adds residual
    
# Implement encoder later
class EncoderLayer(Layer):  # Defines a custom Keras layer representing a transformer encoder block.
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):  # Initializes the encoder layer with embedding size, number of heads, feed-forward size, and dropout rate.
        super(EncoderLayer, self).__init__()  # Calls the parent Layer class constructor.
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)  # Multi-head self-attention layer to let the model focus on different parts of the input.
        self.ffn = tf.keras.Sequential([  # Feed-forward neural network for further processing after attention.
            Dense(ff_dim, activation="relu"),  # First dense layer with ReLU activation for non-linearity.
            Dense(embed_dim),  # Second dense layer projecting back to the embedding dimension.
        ]) 
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # Layer normalization after attention and residual connection.
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # Layer normalization after feed-forward and residual connection.
        self.dropout1 = Dropout(rate)  # Dropout layer after attention for regularization.
        self.dropout2 = Dropout(rate)  # Dropout layer after feed-forward for regularization.
    def call(self, inputs, training):  # Defines the forward pass for the encoder layer.
        attn_output = self.att(inputs)  # Applies multi-head self-attention to the inputs.
        attn_output = self.dropout1(attn_output, training=training)  # Applies dropout to the attention output.
        out1 = self.layernorm1(inputs + attn_output)  # Adds residual connection and normalizes.
        ffn_output = self.ffn(out1)  # Passes through the feed-forward network.
        ffn_output = self.dropout2(ffn_output, training=training)  # Applies dropout to the feed-forward output.
        return self.layernorm2(out1 + ffn_output)  # Adds residual connection and normalizes again, then returns the output.

# Implement Transformer Encoder
class TransformerEncoder(Layer): 
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1): 
        super(TransformerEncoder, self).__init__() 
        self.num_layers = num_layers 
        self.embed_dim = embed_dim 
        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)] 
        self.dropout = Dropout(rate) 

    def call(self, inputs, training=False): 
        x = inputs 
        for i in range(self.num_layers): 
            x = self.enc_layers[i](x, training=training) 
        return x 
# Example usage 
embed_dim = 128 
num_heads = 8 
ff_dim = 512 
num_layers = 4 
transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim) 
inputs = tf.random.uniform((1, 100, embed_dim)) 
outputs = transformer_encoder(inputs, training=False)  # Use keyword argument for 'training' 
print(outputs.shape)  # Should print (1, 100, 128) 

# Build and compile the model
# Build the model 
input_shape = (X.shape[1], X.shape[2]) 
inputs = tf.keras.Input(shape=input_shape) 
# Project the inputs to the embed_dim 
x = tf.keras.layers.Dense(embed_dim)(inputs) 
encoder_outputs = transformer_encoder(x) 
flatten = tf.keras.layers.Flatten()(encoder_outputs) 
outputs = tf.keras.layers.Dense(1)(flatten) 
model = tf.keras.Model(inputs, outputs) 
# Compile the model 
model.compile(optimizer='adam', loss='mse') 
# Summary of the model, uncomment the next line to see the model architecture 
# print(model.summary())

# Train the model, takes about 10-15 mins on CPU
model.fit(X, Y, epochs=20, batch_size=32)

# Make predictions 
predictions = model.predict(X) 
predictions = scaler.inverse_transform(predictions) 
# Prepare true values for comparison
true_values = scaler.inverse_transform(data.reshape(-1, 1))
# Plot the predictions vs true values
import matplotlib.pyplot as plt 
plt.plot(true_values, label='True Data') 
plt.plot(np.arange(time_step, time_step + len(predictions)), predictions, label='Predictions') 
plt.xlabel('Time') 
plt.ylabel('Stock Price') 
plt.legend() 
plt.title('Predictions vs True Data (Both Scaled Back)')
# Uncomment the next line to display the plot
#plt.show() 
