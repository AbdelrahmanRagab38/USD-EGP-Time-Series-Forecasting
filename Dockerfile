# Specify a base image
FROM python:3.9.9

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the app files
COPY . .

# Expose the port
EXPOSE 5000

# Start the app
CMD [ "python", "app.py" ]
