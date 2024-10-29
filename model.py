import torch, torchvision
def create_vitB16_model(num_classes: int=3, seeds: int = 42):

  # 1. Setup pretrained viT Weights
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT

  # 2. Get transforms
  transforms = weights.transforms()

  # 3. Setup pretrained instance
  model = torchvision.models.vit_b_16(weights=weights)

  # 4. Freeze the base layers in the model (this will stop all layers from training)
  for params in model.parameters():
    params.requires_grad = False

  # Set seeds for reproducibility
  torch.manual_seed(seeds)

  # 5. Modify the number of output layers
  model.heads = torch.nn.Sequential(
      torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
  )

  return model, transforms
