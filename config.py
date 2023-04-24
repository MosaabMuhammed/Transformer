class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    max_len = 256
    d_model = 512
    n_layers = 6
    n_heads = 8
    ffn_hidden = 2048
    
    # Optimizer parameter setting
    init_lr = 3e-4
    factor = 0.9
    adam_eps = 5e-9
    patience = 10
    warmup = 100
    n_epochs = 1000
    clip = 1.0
    weight_decay = 5e-4
    inf = float('inf')