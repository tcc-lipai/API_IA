from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import difflib
import re
from transformers import Wav2Vec2ForCTC, AutoProcessor
import shutil
import os

# Inicializa a API
app = FastAPI(
    title="API de Avaliação de Pronúncia",
    version="1.0.0"
)

# Configura o CORS para permitir conexões com o React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo de IA apenas UMA vez ao iniciar o servidor
MODEL_ID = "facebook/mms-1b-all"
print("Carregando o modelo MMS da Meta (isso pode demorar na primeira execução)...")

# Carrega os componentes base globais
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

print("API e Modelo de IA carregados com sucesso!")

def limpar_texto(texto):
    """Padroniza o texto para minúsculas e remove pontuações e espaços sobressalentes."""
    if not texto:
        return ""
    texto = texto.lower()
    # Remove pontos, interrogações, exclamações, vírgulas e traços comuns em textos
    texto = re.sub(r'[.\!?,\-\(\)]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()

@app.post("/verificar-pronuncia/", summary="Verifica a pronúncia de um áudio")
async def verificar_pronuncia(
    texto_esperado: str = Form(..., description="A palavra ou frase que a criança deveria falar"),
    arquivo: UploadFile = File(..., description="O arquivo de áudio capturado (.wav ou .mp3)")
):
    # Cria um nome temporário seguro para salvar o arquivo enviado pelo usuário
    temp_path = f"temp_{arquivo.filename}"
    
    # Salva o arquivo em disco temporariamente
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(arquivo.file, buffer)
    
    try:
        # Configura o idioma do tokenizador especificamente nesta execução (Evita bugs no Swagger)
        processor.tokenizer.set_target_lang("por")
        model.load_adapter("por")

        # 1. Carrega o áudio e converte para a taxa de amostragem exigida pela IA (16kHz)
        audio, sr = librosa.load(temp_path, sr=16000)
        
        # 2. Prepara os dados para a IA
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # 3. Faz a transcrição (Inferência)
        with torch.no_grad():
            logits = model(**inputs).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        texto_transcrito = processor.decode(predicted_ids[0])
        
        # 4. Limpa e padroniza os textos antes da comparação
        original = limpar_texto(texto_esperado)
        falado = limpar_texto(texto_transcrito)
        
        # 5. Compara as strings e calcula a similaridade
        similaridade = difflib.SequenceMatcher(None, original, falado).ratio()
        porcentagem_acerto = round(similaridade * 100, 2)
        
        # Sistema de feedback lúdico adaptado para o público infantil (Gamificação para o TCC)
        if porcentagem_acerto >= 85:
            feedback = "Incrível! Você arrasou! "
        elif porcentagem_acerto >= 60:
            feedback = "Muito bem! Quase lá, vamos tentar de novo?"
        else:
            feedback = "Precisa treinar mais! Continue tentando, não desista!"

        return {
            "sucesso": True,
            "texto_esperado": texto_esperado,
            "texto_falado": texto_transcrito if texto_transcrito else "[Áudio não reconhecido]",
            "porcentagem_acerto": porcentagem_acerto,
            "feedback": feedback
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar áudio: {str(e)}")
        
    finally:
        # Garante que o arquivo temporário será deletado mesmo se houver erro interno
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/", summary="Verifica o status da API")
def home():
    return {"status": "API Online", "modelo": MODEL_ID}