
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (EMBEDDING_DIM, N_WAY, N_SUPPORT, 
                   PROTO_WEIGHT, RELATION_WEIGHT, TEMPERATURE, EPISODES)
class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, input_dim, num_heads = 4):
        super().__init__()
        self.num_heads = num_heads

        # ensure input_Dim is divisible by num_heads
        assert input_dim % num_heads == 0, "input dimension should be divisible by number of attention heads"
        self.head_dim = input_dim // num_heads

        # create query, key, value projections for each head
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)

        # output projection
        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        # input x = [batch_size, seq_len, input_dim]
        # project into query, key, values
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # transpose for attention computation [batch, heads, seq_len, head_dim]

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # compute attention scores, how much each token should focus on each other token (matrix of attention scores)
        scores = torch.matmul(q, k.transpose(-1,-2))/ (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim = -1) # softmax across heads

        # apply attention to values
        context = torch.matmul(attention, v)

        # reshape context 
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, input_dim)  
        # context -> [batch, seq-len, input_dim]
        # apply output projection
        output = self.output_linear(context)

        # weighted sum over sequence dimension
        weights = F.softmax(torch.sum(output, dim = -1, keepdim = True), dim = 1)
        final_context = torch.sum(output * weights, dim  = 1)

        return final_context

class LearnableNormalization(nn.Module):
    # direction of vector matters in cosine sim, so it centers feature 
    def __init__(self, embedding_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embedding_dim)) # NORMALIZED embedding dimension acorss batch
        self.bias = nn.Parameter(torch.zeros(embedding_dim)) # learnable parameters
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        # apply batch normalization
        if x.dim() == 2:
            x = self.bn(x)
        else:
            batch_size = x.size(0)
            x = self.bn(x.view(batch_size, -1)).view(x.shape) # ensure embeddings say centered 

        # apply learnable scaling
        return x * self.scale + self.bias 

class CombinedFreqTemporalCNNEncoder(nn.Module):
    # Frequency attention and temporal structure encoder
    def __init__(self, n_classes=N_WAY, rnn_type = 'gru', bidirectional = True, hidden_size = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # preserve time dimension for temporal modelling, so we use adaptive pooling only on frequency dimensions
        self.time_preserve_pool = nn.AdaptiveAvgPool2d((4, None))

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if rnn_type.lower() == 'lstm': # can try
            self.rnn = nn.LSTM(
                input_size = 128 * 4, # 128 channels * 4 frequency bins
                hidden_size = hidden_size,
                batch_first = True,
                bidirectional = bidirectional
        )
        else:  # Default to GRU -> very high accuracy
            self.rnn = nn.GRU(
                input_size=128 * 4,  # 128 channels x 4 frequency bins
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional
            )

        # attention mechanism - focus on the important tiem  step
        self.attention = MultiHeadTemporalAttention(
            input_dim = hidden_size * self.num_directions, num_heads=4
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_size * self.num_directions, EMBEDDING_DIM)
        self.embedding_norm = LearnableNormalization(EMBEDDING_DIM)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(EMBEDDING_DIM, n_classes)  # Classification layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    
    def forward(self, input_data, return_embedding = True):
        # Handle multiple possible input tensor shapes
        
        # If input is 6D, try to reshape
        if input_data.dim() == 6:
            # Reshape from [batch, k_shot, 1, 1, height, width] to [batch * k_shot, 1, height, width]
            batch_size, k_shot, channels, _, height, width = input_data.shape
            input_data = input_data.view(batch_size * k_shot, channels, height, width)
        
        # Handle 5D input tensor 
        elif input_data.dim() == 5:
            # Reshape from [batch, n_samples, channels, height, width] to [batch * n_samples, channels, height, width]
            batch_size, n_samples, channels, height, width = input_data.shape
            input_data = input_data.view(batch_size * n_samples, channels, height, width)
        
        # Ensure input has proper dimensions for spectrogram
        # Check if batch dimension exists, if not add it
        if input_data.dim() == 2:  # Single spectrogram with shape [n_mels, time]
            input_data = input_data.unsqueeze(0)  # Add batch dimension: [1, n_mels, time]
            input_data = input_data.unsqueeze(1)  # Add channel dimension: [1, 1, n_mels, time]
        elif input_data.dim() == 3:
            # This could be [batch, n_mels, time] or [1, n_mels, time]
            if input_data.size(0) == 1 and len(input_data) == 1:
                # It's likely [1, n_mels, time], add channel dim
                input_data = input_data.unsqueeze(1)
            else:
                # It's likely [batch, n_mels, time], add channel dim to each
                input_data = input_data.unsqueeze(1)
        
        # Ensure input is 4D [batch, channels, height, width]
        if input_data.dim() != 4:
            raise ValueError(f"Unexpected input tensor shape: {input_data.shape}")
        
        batch_size = input_data.size(0)  # remember original batch size for shaping

        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Preserve the time dimension by only pooling frequency
        x = self.time_preserve_pool(x) # [batch, 128, 4, T]

        # reshape for rnn [batch, timesteps, features]
        # transpose to get time as the second dimension (in x)
        x = x.permute(0, 3, 1, 2 ) # -> 0,3,1,2 = order
        time_steps = x.size(1)
        x = x.reshape(batch_size, time_steps, -1) #[batch, T, 128*4]

        # process with rnn
        rnn_out, _ = self.rnn(x) #[batch, T, hidden_size*num_directions]

        # get context vector with attention weight
        context = self.attention(rnn_out)  # This should return [batch, hidden_size * num_directions]
        
        # Get embeddings
        embedding = self.fc1(context)

        normalized_embedding = self.embedding_norm(embedding)
        
        # Return embeddings if requested
        if return_embedding:
            return normalized_embedding
        
        # Otherwise continue with classification
        x = self.dropout(embedding)
        logits = self.fc2(x)
        return self.log_softmax(logits)


class PrototypicalNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.initial_temperature = 10.0
        self.final_temperature = 3.0
        self.current_temperature = self.initial_temperature
        self.total_episodes = EPISODES    
        self.episode_counter = 0
    def update_temperature(self):
        # Linear decay, reducing temperature across episodes to encourage model to be more confident and sharpen softmax 
        self.episode_counter += 1
        progress = min(1.0, self.episode_counter / self.total_episodes)
        self.current_temperature = self.initial_temperature - progress * (self.initial_temperature - self.final_temperature)
    def forward(self, support_images, support_labels, query_images, n_way, n_support):
        self.update_temperature()
        """
        Implements the prototypical network for classification
        
        Parameters:
            support_images: [n_way*n_support, C, H, W] or [n_way*n_support, H, W] support images
            support_labels: [n_way*n_support] support labels
            query_images: [n_query, C, H, W] or [n_query, H, W] query images
            n_way: number of classes
            n_support: number of examples per class in support set
            
        Returns:
            log_p_y: [n_query, n_way] log probabilities for each query
        """
        # Extract feature embeddings for support and query sets
        support_embeddings = self.encoder(support_images, return_embedding=True)
        query_embeddings = self.encoder(query_images, return_embedding=True)
        
        # Get unique classes
        unique_labels = torch.unique(support_labels)
        
        # Ensure we have the right number of classes
        if len(unique_labels) != n_way:
            raise ValueError(f"Expected {n_way} unique classes but got {len(unique_labels)}")
        
        # Compute prototypes
        prototypes = torch.zeros(n_way, support_embeddings.shape[1], device=support_embeddings.device)
        for i, label in enumerate(unique_labels):
            mask = support_labels == label
            prototypes[i] = support_embeddings[mask].mean(dim=0)

        query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
        # Calculate cosine similarity with temperature scaling
        cosine_sim = torch.mm(query_embeddings_norm, prototypes_norm.t()) * TEMPERATURE

        dists = 1 - cosine_sim  # convert similarity to distance
        # Convert distances to log probabilities
        log_p_y = F.log_softmax(-dists, dim=1)
        
        return log_p_y
    
    def classify(self, support_images, support_labels, query_images, n_way, n_support):
        """
        Perform classification using prototypical network
        
        Returns:
            predicted_labels: [n_query] predicted class indices for each query
        """
        log_p_y = self.forward(support_images, support_labels, query_images, n_way, n_support)
        _, predicted_labels = torch.max(log_p_y, dim=1)
        return predicted_labels


class EnsembleModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.proto_net = PrototypicalNet(encoder)

    
    def forward(self, x):
        """
        Standard forward method for typical classification
        
        Parameters:
            x: input images [batch, channels, height, width]
        
        Returns:
            class probabilities
        """
        # Use encoder's standard forward method
        return self.encoder(x)
    
    def few_shot_classify(self, support_images, support_labels, query_images, 
                       n_way=N_WAY, n_support=N_SUPPORT,
                       proto_weight=PROTO_WEIGHT, relation_weight=RELATION_WEIGHT):
        """
        Few-shot classification method that matches the previous implementation
        
        Parameters:
            support_images: [n_way*n_support, C, H, W] support 
            support_labels: [n_way*n_support] support labels
            query_images: [n_query, C, H, W] query images
            n_way: number of classes
            n_support: number of examples per class in support set
            proto_weight: weight for prototypical network predictions
            relation_weight: weight for relation network predictions
            
        Returns:
            predicted_labels: [n_query] predicted class indices for each query
        """
        proto_log_probs = self.proto_net.forward(support_images, support_labels, query_images, n_way,n_support)
        proto_probs = torch.exp(proto_log_probs)
        
        # Return predicted labels
        _, predicted_labels = torch.max(proto_probs, dim=1)
        return predicted_labels