import os

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
REDIS_URL       = os.getenv("REDIS_URL",       "redis://localhost:6379")
DB_URL          = os.getenv("DB_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/fraud_detection"
)