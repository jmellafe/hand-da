# Build the Docker image
docker build -t hand_da .

# Run the Docker container, linking the current folder with /app
docker run --gpus all -it -v "$(pwd)":/app -v ~/.aws:/root/.aws hand_da
