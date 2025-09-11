# This builds a basic transformer model for text generation
# Install required libraries
# python3 -m pip install pandas matplotlib tensorflow scikit-learn
import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import TextVectorization 
from tensorflow.keras.utils import get_file 

# Load the dataset, shakespeare.txt, from a URL
path_to_file = get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt') 
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') 
# Preview the dataset 
# uncomment the following line to see the first 1000 characters of the text
# print(text[:1000]) 

# Preprocess the dataset 
vocab_size = 10000 
seq_length = 100 
# Adapt TextVectorization to full text 
vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int') 
text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1) 
vectorizer.adapt(text_ds) 
# Vectorize the text 
vectorized_text = vectorizer([text])[0] 
# uncomment the following lines to see the shape and first 10 tokens of the vectorized text
# print("Vectorized text shape:", vectorized_text.shape) 
# print("First 10 vectorized tokens:", vectorized_text.numpy()[:10]) 

# Create input-target sequences
def create_sequences(text, seq_length):  # Defines a function to create input and target sequences from text.
    input_seqs = []  # Initializes a list to store input sequences.
    target_seqs = []  # Initializes a list to store target sequences.
    for i in range(len(text) - seq_length):  # Loops over the text to generate sequences of the given length.
        input_seq = text[i:i + seq_length]  # Extracts a sequence of length seq_length as input.
        target_seq = text[i + 1:i + seq_length + 1]  # Extracts the next sequence as the target (shifted by one).
        input_seqs.append(input_seq)  # Adds the input sequence to the list.
        target_seqs.append(target_seq)  # Adds the target sequence to the list.
    return np.array(input_seqs), np.array(target_seqs)  # Converts lists to numpy arrays and returns them.
# Generate sequences 
X, Y = create_sequences(vectorized_text.numpy(), seq_length) 
# Check if sequences are correctly generated, uncomment the following lines to see the number of sequences and a sample 
# print("Number of sequences generated:", len(X)) 
# print("Sample input sequence:", X[0] if len(X) > 0 else "No sequences generated") 
# Check if X and Y are not empty 
assert X.size > 0, "Input data X is empty" 
assert Y.size > 0, "Target data Y is empty" 
X = tf.convert_to_tensor(X) 
Y = tf.convert_to_tensor(Y) 
# uncomment the following lines to see the shapes of X and Y
# print("Shape of X:", X.shape) 
# print("Shape of Y:", Y.shape)

from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout  
from tensorflow.keras.models import Model  

class TransformerBlock(tf.keras.layers.Layer):  # Defines a custom Keras layer for a transformer block.
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):  # Initializes the transformer block with embedding size, number of heads, feed-forward size, and dropout rate.
        super(TransformerBlock, self).__init__()  # Calls the parent Layer class constructor.
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)  # Multi-head attention layer to let the model focus on different parts of the input.
        self.ffn = tf.keras.Sequential([  # Feed-forward neural network for further processing after attention.
            Dense(ff_dim, activation="relu"),  # First dense layer with ReLU activation for non-linearity.
            Dense(embed_dim),  # Second dense layer projecting back to the embedding dimension.
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # Layer normalization after attention and residual connection.
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # Layer normalization after feed-forward and residual connection.
        self.dropout1 = Dropout(rate)  # Dropout layer after attention for regularization.
        self.dropout2 = Dropout(rate)  # Dropout layer after feed-forward for regularization.

    def call(self, inputs, training=False):  # Defines the forward pass for the transformer block.
        attn_output = self.att(inputs, inputs)  # Applies multi-head self-attention to the inputs.
        attn_output = self.dropout1(attn_output, training=training)  # Applies dropout to the attention output.
        out1 = self.layernorm1(inputs + attn_output)  # Adds residual connection and normalizes.
        ffn_output = self.ffn(out1)  # Passes through the feed-forward network.
        ffn_output = self.dropout2(ffn_output, training=training)  # Applies dropout to the feed-forward output.
        return self.layernorm2(out1 + ffn_output)  # Adds residual connection and normalizes again, then returns the output.

class TransformerModel(Model):  # Defines a custom Keras model for the transformer.
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):  # Initializes the transformer model with vocabulary size, embedding size, number of heads, feed-forward size, number of layers, and sequence length.
        super(TransformerModel, self).__init__()  # Calls the parent Model class constructor.
        self.embedding = Embedding(vocab_size, embed_dim)  # Embedding layer to convert token indices to dense vectors.
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)  # Computes positional encoding for input sequences.
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]  # Creates a list of transformer blocks.
        self.dense = Dense(vocab_size)  # Final dense layer to project outputs to vocabulary size for prediction.

    def positional_encoding(self, seq_length, embed_dim):  # Computes positional encoding for the input sequence.
        angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)  # Calculates angle rates for each position and dimension.
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Applies sine to even indices in the array.
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Applies cosine to odd indices in the array.
        pos_encoding = angle_rads[np.newaxis, ...]  # Adds a batch dimension to the positional encoding.
        return tf.cast(pos_encoding, dtype=tf.float32)  # Returns the positional encoding as a float32 tensor.

    def get_angles(self, pos, i, embed_dim):  # Helper function to compute the angles for positional encoding.
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))  # Calculates the angle rates for each position and dimension.
        return pos * angle_rates  # Returns the product of positions and angle rates.

    def call(self, inputs, training=False):  # Defines the forward pass for the transformer model.
        seq_len = tf.shape(inputs)[1]  # Gets the sequence length from the input tensor.
        x = self.embedding(inputs)  # Applies the embedding layer to the inputs.
        x += self.pos_encoding[:, :seq_len, :]  # Adds positional encoding to the embeddings.
        for transformer_block in self.transformer_blocks:  # Loops through each transformer block.
            x = transformer_block(x, training=training)  # Passes the data through the transformer block.
        output = self.dense(x)  # Applies the final dense layer to get predictions for each token.
        return output  # Returns
    
# Define hyperparameters 
embed_dim = 256 
num_heads = 4 
ff_dim = 512 
num_layers = 4 
# Build the Transformer model 
model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length)
# Provide input shape to build the model by passing a dummy input with maxval specified
_ = model(tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32))
# Compile the model 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# Summary of the model, uncomment the following line to see the model architecture 
#model.summary()

# Since the darta is large, we will use a subset for training
X = X[:10000]
Y = Y[:10000]
# Import necessary libraries for training visualization
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
# Early stopping callback to stop training if the loss doesn't improve
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
# Train the transformer model on the full input and target sequences, 2 epochs should be sufficient for demo purposes
# This will take a long time to run on a CPU, ideally use a GPU
history = model.fit(X, Y, epochs=2, batch_size=32, callbacks=[early_stopping])
# Plot training loss to monitor model performance over epochs
# Uncomment the following lines to see the training loss plot
# plt.plot(history.history['loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

# Trained model can now be used for text generation tasks
# Define a function to generate text using the trained model
def generate_text(model, start_string, num_generate=100, temperature=1.0):
    # Convert the start string to a vectorized format
    input_eval = vectorizer([start_string]).numpy()
    # Ensure the input length is the same as the model's expected input shape
    if input_eval.shape[1] < seq_length:
        # Pad the input if it's shorter than the expected sequence length
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
    elif input_eval.shape[1] > seq_length:
        # Truncate the input if it's longer than the expected sequence length
        input_eval = input_eval[:, -seq_length:]
    input_eval = tf.convert_to_tensor(input_eval)
    # Initialize an empty list to store generated text
    text_generated = []
    # Start generating text
    for i in range(num_generate):
        # Make predictions using the model
        predictions = model(input_eval)
        # Remove only the batch dimension, keep the logits as 2D (batch_size, vocab_size)
        predictions = predictions[0]  # This should be of shape [vocab_size]
        # Apply temperature to predictions
        predictions = predictions / temperature    
        # Use a categorical distribution to predict the next word
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        # Update the input tensor to include the predicted word, maintaining the sequence length
        input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)  # Append predicted token
        input_eval = input_eval[:, -seq_length:]  # Keep only the last `seq_length` tokens
        input_eval = tf.convert_to_tensor(input_eval)  # Convert back to tensor
        # Append the predicted word to the generated text
        text_generated.append(vectorizer.get_vocabulary()[predicted_id])
    # Return the generated text starting from the initial seed
    return start_string + ' ' + ' '.join(text_generated)
# Generate text with temperature control
start_string = "To be, or not to be"
generated_text = generate_text(model, start_string, temperature=0.7)  # Lower temperature for more focused predictions
print(generated_text)