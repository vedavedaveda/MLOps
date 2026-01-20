import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
