# citizen-shield-dash
A dashboard used to upload a csv and to run the analyses used in the paper.

## Installation
- `git clone git@github.com:Citizen-Shield/citizen-shield-dash.git` # Clone the repository
- `cd citizen-shield-dash` # Change directory
- `python3 -m venv venv` # Create a virtual environment
- `source venv/bin/activate` # Activate the virtual environment (`deactivate` to deactivate)
- `pip install -r requirements.txt` # Install the requirements
- `streamlit run app.py` # Run the app

## Docker
- `docker build -t citizen-shield-dash .` # Build the image
- `docker run -p 8501:8501 citizen-shield-dash` # Run the container