import base64
import json
import os
import random

import boto3
from tqdm.auto import tqdm

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Image Generator G1.
model_id = "amazon.titan-image-generator-v1"

# Define the image generation prompts for the model.
prompts1 = [
    "A breathtaking view of a mountain range during sunrise.",
    "A tranquil beach with crystal clear water and palm trees.",
    "A lush green forest in the heart of spring.",
    "A vibrant autumn landscape with colorful leaves.",
    "A serene lake reflecting the surrounding hills at dusk.",
    "A picturesque valley filled with wildflowers in bloom.",
    "A dramatic desert landscape with sand dunes at sunset.",
    "A peaceful countryside with rolling hills and a winding river.",
    "A stunning coastal cliff overlooking the ocean waves.",
    "A magical snowy landscape with pine trees and a cabin."
]
prompts2 = [
    "A serene landscape of a misty forest at sunrise, with golden light filtering through the trees and a calm river flowing in the foreground, ultra-realistic and soft lighting",
    "A futuristic cityscape at night, with glowing neon lights reflecting on wet streets, flying cars and towering skyscrapers, cyberpunk style, highly detailed",
    "A majestic lion standing proudly on a cliff at sunset, with a dramatic orange sky and rolling hills in the background, hyper-realistic, high detail fur texture",
    "An abstract painting of swirling vibrant colors, reminiscent of Van Gogh's 'Starry Night', using bold brushstrokes and a mix of blue, yellow, and white",
    "A beautiful, tranquil Japanese garden with a koi pond, cherry blossom trees in full bloom, and a traditional tea house, soft sunlight filtering through the branches",
    "A fantasy scene of a dragon flying over a medieval castle, with smoke rising from its nostrils and a stormy sky in the background, highly detailed, dark fantasy style",
    "A close-up of a dew-covered spiderweb in the morning, with sunlight sparkling on the droplets, extremely detailed, sharp focus on the texture and reflection",
    "A peaceful 1920s Parisian street view, featuring cozy outdoor cafes, charming cobblestone pathways, and vintage buildings with intricate architecture.",
    "An astronaut standing on the surface of Mars, gazing at the Earth in the distance, with red rocky terrain and a clear blue sky, photorealistic, high contrast",
    "A magical winter wonderland with snow-covered trees, a frozen lake reflecting the pale blue sky, and soft sunlight peeking through the branches, ultra-realistic and serene."
]


# Number of images per prompt
images_per_prompt = 500

# Output directory
output_dir = ""
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Batch generation
for pi,prompt in enumerate(prompts2):
    with tqdm(total=images_per_prompt,desc=f"Generating images for prompt '{pi}'") as pbar:
        for i in range(images_per_prompt):
            # Generate a random seed.
            seed = random.randint(0, 2147483647)
            image_path = os.path.join(output_dir, f"p{pi}_{i:05d}.png")
            if os.path.exists(image_path):
                pbar.update(1)
                continue
            # Format the request payload using the model's native structure.
            native_request = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "quality": "standard",
                    "cfgScale": 8.0,
                    "height": 512,
                    "width": 512,
                    "seed": seed,
                },
            }

            # Convert the native request to JSON.
            request = json.dumps(native_request)

            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_id, body=request)

            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract the image data.
            base64_image_data = model_response["images"][0]

            # Save the generated image to a local folder.
            image_data = base64.b64decode(base64_image_data)

            with open(image_path, "wb") as file:
                file.write(image_data)

            print(f"The generated image for prompt '{prompt}' has been saved to {image_path}")
            pbar.update(1)
