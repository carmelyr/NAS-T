import torch
from model_builder import build_model

# Dummy input
batch_size = 8
sequence_length = 5
input_dim = 10
num_classes = 2

X = torch.randn(batch_size, sequence_length, input_dim)
y = torch.randint(0, 2, (batch_size,))
y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

# Try building and running the transformer
try:
    model = build_model("Transformer", input_size=input_dim, num_heads=1, num_layers=1, output_size=num_classes)
    out = model(X)
    print("Transformer output:", out.shape)
except Exception as e:
    import traceback
    print("Transformer test failed:")
    traceback.print_exc()
