'''
This script defines a transformation pipeline for image preprocessing using torchvision.
It includes resizing, random horizontal flipping, random rotation, conversion to tensor,
and normalization.
'''
test_transform = transforms.Compose([transforms.Resize((128, 128)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ToTensor(), # Converts PIL Image to a tensor and scales values to [0, 1]
                                     transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # Normalize to [-1, 1]
])

'''
Other transformations can be added as needed, such as:
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))
'''