# API_LIPAI - Passos
Entrar no Prompt de Comando

python -m venv venv

.\venv\Scripts\activate

python -m pip install torch transformers librosa accelerate python-multipart fastapi uvicorn soundfile

pip freeze > requirements.txt

uvicorn script:app --reload


http://127.0.0.1:8000/docs
