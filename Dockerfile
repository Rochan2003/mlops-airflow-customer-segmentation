# The official Airflow image already ships pandas, numpy, scikit-learn, joblib
# and scipy. Only kneed is missing, so this layer is deliberately small.
FROM apache/airflow:3.3.2

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
