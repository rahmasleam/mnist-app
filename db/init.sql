CREATE TABLE IF NOT EXISTS "Prediction_History" (
    id SERIAL PRIMARY KEY,
    "Timestamp" TIMESTAMP,
    "Pred" INT,
    "True" INT,
    "Conf" FLOAT
);
