# The following code will only execute
# successfully when compression is complete

import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "sahamed/lsun-bedroom",
    output_dir='datasets/lsun_bedrooms')

print("Path to dataset files:", path)