# Animalbot Chatbot

This is the chatbot component of the Animalbot project, which simulates conversations as either a duck or a fox based on user interaction.

## Features

- Dynamically switches between duck and fox personalities
- Uses LLMs from chat-ai.academiccloud.de
- Logs conversation history and classification details

## Running Locally

To run the chatbot locally without Docker:

1. Make sure you have Python 3.8+ installed
2. Create a `.env` file in the project root with your API key
3. Run:
   ```bash
   cd ..  # Go to project root
   python -m chatbot.animalbot
   ```

For Docker-based deployment, see the main project README.

## Helpful Commands

### Docker Compose Commands

Run a single instance with default settings:
```bash
docker compose -p animalbot up -d
```

Run multiple instances with custom ports:
```bash
# Instance 1
API_PORT=8003 UI_PORT=8504 API_BASE_URL=http://localhost docker compose -p animalbot1 up -d

# Instance 2
API_PORT=8005 UI_PORT=8506 API_BASE_URL=http://localhost docker compose -p animalbot2 up -d
```

Use current directory name as project name:
```bash
API_PORT=8003 UI_PORT=8504 API_BASE_URL=http://localhost docker compose -p $(basename "$PWD") up -d
```

### Viewing Logs

View logs from a specific instance:
```bash
docker logs $(docker ps -qf "name=animalbot_animalbot")
```

Follow logs in real-time:
```bash
docker logs -f $(docker ps -qf "name=animalbot_animalbot")
```

### Managing Containers

Stop a specific instance:
```bash
docker compose -p animalbot1 down
```

Rebuild and restart after code changes:
```bash
docker compose -p animalbot build --no-cache
docker compose -p animalbot up -d
```

List all running instances:
```bash
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"
```