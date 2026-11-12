import streamlit as st
import streamlit.components.v1 as components
from groq import Groq  # <-- CAMBIO: Reemplazamos ollama por groq
from gtts import gTTS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from pathlib import Path
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import unicodedata

BASE = Path(__file__).resolve().parent


def _leer_curso(carpeta: Path) -> dict:
    """curso.json de cada curso define su nombre, público y vocabulario. Campos opcionales: boton_extra {etiqueta, pregunta}."""
    por_defecto = {"titulo": carpeta.name, "descripcion": "una capacitación de Seguridad y Salud en el Trabajo",
                   "publico": "personal operativo", "vocabulario": "SST, EPP, peligro, riesgo, medidas de control"}
    try:
        with open(carpeta / "curso.json", encoding="utf-8") as f:
            por_defecto.update(json.load(f))
    except Exception:
        pass
    return por_defecto


st.set_page_config(page_title="Aula Virtual SST", layout="wide", page_icon="🎓")

# =======================================================
# CONFIGURACIÓN
# =======================================================
CARPETA_CURSOS = BASE / "cursos"         # una subcarpeta por curso: curso.json + guiones.json + laminas/
CARPETA_AUDIO = BASE / "audio_cache"     # el audio de cada lámina se guarda aquí (se genera 1 sola vez)

# --- CONFIGURACIÓN DE GROQ ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY_RUTAS") # Lee la llave desde los secretos de GitHub Codespaces 
MODELO = "openai/gpt-oss-20b" # Modelo de Groq. También puedes usar "mixtral-8x7b-32768" 

STT_MOTOR = "whisper"      # voz -> texto: "whisper" (offline) | "google" (online)
WHISPER_MODELO = "small"    # "base" (rápido, menos preciso) | "small" (recomendado) | "medium" (lento en CPU)
TTS_MOTOR = "edge"         # texto -> voz: "edge" (online, natural) | "gtts" | "pyttsx3" (offline)
VOZ_EDGE = "es-PE-AlexNeural"  # otras: "es-PE-CamilaNeural", "es-MX-JorgeNeural", "es-ES-AlvaroNeural"
VELOCIDAD_EDGE = "-5%"     # más pausado, tipo clase
PAUSA_ENTRE_LAMINAS_MS = 800   # pausa tras terminar el audio, antes de pasar sola a la siguiente lámina

CARPETA_AUDIO.mkdir(exist_ok=True)


# =======================================================
# CARGA DE CONTENIDO
# =======================================================
def _clave_natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


def descubrir_cursos() -> dict:
    """Cada subcarpeta de cursos/ con guiones.json y laminas/ es un curso. Si no hay cursos/, la carpeta de la app es el único curso."""
    cursos = {}
    if CARPETA_CURSOS.is_dir():
        for d in sorted(CARPETA_CURSOS.iterdir(), key=_clave_natural):
            if d.is_dir() and (d / "guiones.json").exists() and (d / "laminas").is_dir():
                cursos[d.name] = d
    if not cursos and (BASE / "guiones.json").exists() and (BASE / "laminas").is_dir():
        cursos[BASE.name] = BASE
    return cursos


@st.cache_data(show_spinner=False)
def cargar_guiones(carpeta: str) -> dict:
    with open(Path(carpeta) / "guiones.json", encoding="utf-8") as f:
        return {int(g["n"]): g for g in json.load(f)}


@st.cache_data(show_spinner=False)
def listar_laminas(carpeta: str) -> list:
    archivos = [p for p in (Path(carpeta) / "laminas").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return sorted(archivos, key=_clave_natural)


CURSOS = descubrir_cursos()
if not CURSOS:
    st.error("No encontré ningún curso. Crea la carpeta `cursos/` y dentro una subcarpeta por curso, "
             "cada una con `guiones.json`, `curso.json` y la carpeta `laminas/`.")
    st.stop()

TITULOS = {k: _leer_curso(p)["titulo"] for k, p in CURSOS.items()}
if st.session_state.get("curso_id") not in CURSOS:
    st.session_state.curso_id = next(iter(CURSOS))
st.session_state.setdefault("selector_curso", st.session_state.curso_id)

CARPETA_CURSO = CURSOS[st.session_state.curso_id]
CURSO = _leer_curso(CARPETA_CURSO)

try:
    GUIONES = cargar_guiones(str(CARPETA_CURSO))
    LAMINAS = listar_laminas(str(CARPETA_CURSO))
except Exception as e:
    st.error(f"No pude cargar el curso «{CARPETA_CURSO.name}». Revisa `guiones.json` y la carpeta `laminas/`. ({e})")
    st.stop()

TOTAL = len(GUIONES)
TOTAL_IMAGENES = len(LAMINAS)
LAMINAS = LAMINAS[:TOTAL]
if TOTAL_IMAGENES != TOTAL:
    st.warning(f"Hay {TOTAL_IMAGENES} imágenes en `laminas/` pero {TOTAL} guiones en `guiones.json`. Deben coincidir.")


# =======================================================
# AUDIO (TEXTO -> VOZ) con caché en disco
# =======================================================
def limpiar_para_voz(texto: str) -> str:
    return re.sub(r"[\*\#\_\[\]\-\—\•]", "", texto)


async def _edge_async(texto: str) -> bytes:
    import edge_tts
    com = edge_tts.Communicate(texto, VOZ_EDGE, rate=VELOCIDAD_EDGE)
    datos = bytearray()
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            datos.extend(chunk["data"])
    return bytes(datos)


def _tts_pyttsx3(texto: str) -> bytes:
    import pyttsx3
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    motor = pyttsx3.init()
    motor.setProperty("rate", 155)
    motor.save_to_file(texto, tmp.name)
    motor.runAndWait()
    with open(tmp.name, "rb") as f:
        datos = f.read()
    os.remove(tmp.name)
    return datos


def sintetizar(texto: str):
    """Devuelve (bytes, mime) o (None, None). Si falla el motor elegido, intenta gTTS."""
    texto = limpiar_para_voz(texto)
    intentos = [TTS_MOTOR] + [m for m in ("gtts",) if m != TTS_MOTOR]
    for motor in intentos:
        try:
            if motor == "edge":
                return asyncio.run(_edge_async(texto)), "audio/mpeg"
            if motor == "pyttsx3":
                return _tts_pyttsx3(texto), "audio/wav"
            buf = io.BytesIO()
            gTTS(text=texto, lang="es").write_to_fp(buf)
            return buf.getvalue(), "audio/mpeg"
        except Exception:
            continue
    return None, None


def _ruta_cache(texto: str) -> Path:
    ext = "wav" if TTS_MOTOR == "pyttsx3" else "mp3"
    clave = hashlib.md5(f"{TTS_MOTOR}|{VOZ_EDGE}|{VELOCIDAD_EDGE}|{texto}".encode()).hexdigest()
    return CARPETA_AUDIO / f"{clave}.{ext}"


def audio_de_texto(texto: str, usar_cache: bool = True):
    """Con caché en disco: la clase se genera una vez y luego es instantánea."""
    ext = "wav" if TTS_MOTOR == "pyttsx3" else "mp3"
    ruta = _ruta_cache(texto)
    mime = "audio/wav" if ext == "wav" else "audio/mpeg"
    if usar_cache and ruta.exists():
        return ruta.read_bytes(), mime
    datos, mime_real = sintetizar(texto)
    if datos and usar_cache:
        tmp = ruta.with_suffix(f".{threading.get_ident()}.tmp")   # escritura atómica (evita archivos a medias)
        tmp.write_bytes(datos)
        os.replace(tmp, ruta)
    return datos, (mime_real or mime)


@st.cache_resource
def _pendientes() -> set:
    return set()


def precargar_en_segundo_plano(texto: str):
    """Genera el audio de la SIGUIENTE lámina mientras suena la actual, para que el cambio sea instantáneo."""
    pendientes = _pendientes()
    if texto in pendientes or _ruta_cache(texto).exists():
        return

    def tarea():
        try:
            audio_de_texto(texto)
        finally:
            pendientes.discard(texto)

    pendientes.add(texto)
    threading.Thread(target=tarea, daemon=True).start()


def reproductor(datos: bytes, mime: str, autoplay: bool, auto_avanzar: bool):
    """Reproductor HTML. Al terminar el audio, en modo clase pulsa 'Siguiente' automáticamente."""
    b64 = base64.b64encode(datos).decode()
    js_fin = ""
    if auto_avanzar:
        js_fin = """
        a.addEventListener('ended', function () {
            setTimeout(function () {
                const botones = window.parent.document.querySelectorAll('button');
                for (const b of botones) {
                    if (b.innerText.includes('Siguiente') && !b.disabled) { b.click(); break; }
                }
            }, 1200);
        });"""
    js_fin = js_fin.replace("1200", str(PAUSA_ENTRE_LAMINAS_MS))
    html = f"""
    <audio id="a" controls {'autoplay' if autoplay else ''} style="width:100%">
        <source src="data:{mime};base64,{b64}" type="{mime}">
    </audio>
    <script>const a = document.getElementById('a'); {js_fin}</script>
    """
    components.html(html, height=54)


# =======================================================
# VOZ -> TEXTO
# =======================================================
@st.cache_resource(show_spinner="Cargando modelo de voz (solo la primera vez)...")
def cargar_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODELO, device="cpu", compute_type="int8")


def _wav_a_numpy(audio_bytes: bytes):
    """Convierte el WAV del micrófono a un arreglo float32 mono de 16 kHz (así Whisper no necesita PyAV/ffmpeg)."""
    import wave
    import numpy as np
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        canales, ancho, tasa = w.getnchannels(), w.getsampwidth(), w.getframerate()
        crudo = w.readframes(w.getnframes())
    if ancho == 2:
        a = np.frombuffer(crudo, dtype=np.int16).astype(np.float32) / 32768.0
    elif ancho == 4:
        a = np.frombuffer(crudo, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif ancho == 1:
        a = (np.frombuffer(crudo, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Formato WAV no soportado ({ancho * 8} bits)")
    if canales > 1:
        a = a[: len(a) // canales * canales].reshape(-1, canales).mean(axis=1)
    a = _remuestrear(a, tasa, 16000)
    pico = float(np.abs(a).max()) if len(a) else 0.0
    if pico > 0.003:                      # sube el volumen si el micrófono grabó bajito
        a = (a * min(0.9 / pico, 12.0)).astype(np.float32)
    return a


def _remuestrear(a, tasa: int, destino: int = 16000):
    """Baja la frecuencia de muestreo CON filtro pasa-bajos (sin filtro se produce 'aliasing' y Whisper entiende peor)."""
    import numpy as np
    if tasa == destino or len(a) < 2:
        return a
    if tasa > destino:
        fc = 0.45 * destino / tasa                # frecuencia de corte normalizada (~7,2 kHz)
        n = 101
        t = np.arange(n) - (n - 1) / 2
        h = np.sinc(2 * fc * t) * np.hamming(n)
        h /= h.sum()
        a = np.convolve(a, h, mode="same")
    n_out = int(len(a) * destino / tasa)
    return np.interp(np.linspace(0, len(a) - 1, n_out), np.arange(len(a)), a).astype(np.float32)


def _transcribir_google(audio_bytes: bytes):
    rec = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:   # requiere WAV
        audio = rec.record(source)
    return rec.recognize_google(audio, language="es-PE")


def transcribir_audio(audio_bytes: bytes):
    """Whisper offline (con conversión propia del WAV). Si falla, intenta con Google (necesita internet)."""
    if STT_MOTOR == "whisper":
        try:
            audio = _wav_a_numpy(audio_bytes)
            if len(audio) < 8000 or float(abs(audio).max()) < 0.01:      # muy corto o silencio
                st.session_state.error_audio = "El audio es muy corto o casi no se oye (revisa el micrófono)."
                return None
            pista = f"Pregunta de un trabajador de reparto sobre respuesta ante emergencias. Vocabulario: {CURSO['vocabulario']}."
            segmentos, _ = cargar_whisper().transcribe(
                audio, language="es", beam_size=5, initial_prompt=pista,
                condition_on_previous_text=False)
            texto = " ".join(seg.text for seg in segmentos).strip()
            return texto or None
        except Exception as e:
            st.session_state.error_audio = f"Whisper: {e}"
    try:
        texto = _transcribir_google(audio_bytes)
        st.session_state.pop("error_audio", None)
        return texto
    except Exception as e:
        prev = st.session_state.get("error_audio", "")
        st.session_state.error_audio = (prev + " | " if prev else "") + f"Google: {e}"
        return None


# =======================================================
# ASISTENTE DE DUDAS (GROQ API)
# =======================================================
SYSTEM_PROMPT_BASE = """Eres el Profesor Jorge, instructor de Seguridad y Salud en el Trabajo (SST) en Perú.
Dictas {descripcion}, dirigida a: {publico}.

REGLAS ESTRICTAS:
- Responde en español, en texto plano (sin markdown), máximo 4 líneas.
- Responde ÚNICAMENTE con la información de los FRAGMENTOS DE LA CLASE. Copia los datos concretos (números de emergencia, pasos, equipos, plazos, nombres) tal como aparecen ahí.
- NO agregues procedimientos, diagnósticos, medicamentos, números telefónicos, normas ni cifras que no estén en los fragmentos, aunque los conozcas. Esto es seguridad: un dato inventado puede costar una vida.
- Si la pregunta no aparece en los fragmentos, di: "Eso no lo vemos en esta capacitación" y sugiere preguntar sobre el tema de la clase.
- Si la pregunta no se entiende (puede venir de un micrófono), pide amablemente que la repita. No repitas tu respuesta anterior.
- Sé cordial, respetuoso y pedagógico. Termina con una pregunta corta."""

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE.format(descripcion=CURSO["descripcion"], publico=CURSO["publico"])

_STOP = set("""de la el los las un una unos unas y o a en por para con sin que se es son al del lo le les su sus mi me te nos
como mas muy si no ni ya hay esta este esto esa ese eso estos estas cual cuales cuando donde quien quienes porque
debe deben debo puede pueden puedo usar uso usa utiliza utilizan sirve sirven hacer hace hacen ser ver tiene tienen
dime dame explicame explica hazme pregunta lamina clase profesor
emergencia emergencias ruta procedimiento protocolo proceder""".split())


def _norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def _raices(t: str) -> set:
    """Palabras clave simplificadas (sin tildes, sin plural: 'férula' y 'férulas' coinciden)."""
    return {w[:5] for w in re.findall(r"[a-z0-9]+", _norm(t)) if len(w) > 2 and w not in _STOP}


def fragmentos_relevantes(pregunta: str, n_actual: int, k: int = 3) -> list:
    """Busca en TODO el guion las láminas que más se parecen a la pregunta (aunque el alumno esté en otra lámina)."""
    q = _raices(pregunta)
    puntajes = []
    for i, g in GUIONES.items():
        if i == n_actual or not q:
            continue
        titulo = _raices(g["titulo"])
        pts = 3 * len(q & titulo) + len(q & _raices(g["resumen"] + " " + g["guion"]))
        if titulo and titulo <= q:      # la pregunta nombra exactamente el tema de la lámina
            pts += 5
        if pts > 0:
            puntajes.append((-pts, i))
    puntajes.sort()
    return [i for _, i in puntajes[:k]]


def construir_mensajes(pregunta: str, n: int, historial: list) -> list:
    indice = "\n".join(f"Lámina {k}: {g['titulo']}" for k, g in GUIONES.items())
    ids = [n] + fragmentos_relevantes(pregunta, n)
    fragmentos = "\n\n".join(f"[Lámina {i}: {GUIONES[i]['titulo']}]\n{GUIONES[i]['guion']}" for i in ids)
    sistema = (f"{SYSTEM_PROMPT}\n\nTEMARIO DE LA CLASE:\n{indice}\n\n"
               f"FRAGMENTOS DE LA CLASE (única fuente permitida; el alumno está viendo la lámina {n}):\n{fragmentos}")
    msgs = [{"role": "system", "content": sistema}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in historial[-4:]]
    msgs.append({"role": "user", "content": f"Pregunta del alumno: {pregunta}"})
    return msgs


def stream_profesor(pregunta: str, n: int, historial: list):
    if not GROQ_API_KEY:
        yield "Error: Falta configurar la API Key de Groq en los Secrets de Codespaces."
        return

    client = Groq(api_key=os.environ.get("GROQ_API_KEY_RUTAS"))  
    mensajes = construir_mensajes(pregunta, n, historial)
    
    try:
        # Llamada a la API de Groq con streaming
        stream = client.chat.completions.create(
            messages=mensajes,
            model=MODELO,
            temperature=0.1,
            max_tokens=800, 
            stream=True
        )
        
        for chunk in stream:
            contenido = chunk.choices[0].delta.content
            if contenido:
                yield contenido
    except Exception as e:
        yield f"Error al conectar con la IA de Groq: {e}"


def verificar_groq():
    if not GROQ_API_KEY:
        return "Falta la API Key de Groq. Ve a Settings > Secrets and variables > Codespaces y crea el secreto GROQ_API_KEY."
    
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY_RUTAS"))  
        # Petición mínima para comprobar que la llave es válida
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hola"}],
            model=MODELO,
            max_tokens=5
        )
        return None
    except Exception as e:
        return f"No puedo conectar con la API de Groq. Revisa tu API Key o tu conexión a internet. (Error: {e})"


# =======================================================
# ESTADO Y CALLBACKS
# =======================================================
def saludo_inicial(titulo: str = None) -> list:
    titulo = titulo or CURSO["titulo"]
    return [{"role": "assistant",
             "content": f"Hola, soy el Profesor Jorge. Pulsa «Iniciar clase» para comenzar la capacitación «{titulo}», o pregúntame lo que quieras."}]


def init_estado():
    defaults = {
        "slide": 1,
        "selector_slide": 1,
        "slide_anterior": 0,
        "modo_clase": False,       # True = la clase se dicta y avanza sola
        "forzar_narracion": False, # narrar una vez la lámina actual sin activar el modo clase
        "audio_actual": None,      # dict(bytes, mime, auto_avanzar)
        "autoplay_pendiente": False,
        "historial": saludo_inicial(),
        "responder_pendiente": False,
        "groq_ok": False,          # <-- CAMBIO: Reemplazado ollama_ok por groq_ok
        "proyector": False,
        "borrador": None,      # pregunta transcrita por voz, pendiente de confirmar
        "borrador_edit": "",
        "ultima_proy": None,   # (pregunta, respuesta) para mostrar en modo proyector
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def cambiar_slide(delta: int):
    st.session_state.slide = min(max(1, st.session_state.slide + delta), TOTAL)
    st.session_state.selector_slide = st.session_state.slide


def ir_a_slide(n: int):
    st.session_state.slide = int(n)


def iniciar_clase():
    st.session_state.modo_clase = True
    st.session_state.slide_anterior = 0   # fuerza narrar la lámina actual


def pausar_clase():
    st.session_state.modo_clase = False
    st.session_state.audio_actual = None
    st.session_state.autoplay_pendiente = False


def escuchar_lamina():
    st.session_state.forzar_narracion = True
    st.session_state.slide_anterior = 0


def enviar_pregunta(texto: str):
    """Una duda pausa la clase automáticamente; luego se reanuda con «Reanudar clase»."""
    pausar_clase()
    st.session_state.historial.append({"role": "user", "content": texto})
    st.session_state.responder_pendiente = True


def guardar_borrador(texto: str):
    st.session_state.borrador = texto
    st.session_state.borrador_edit = texto


def enviar_borrador():
    texto = (st.session_state.get("borrador_edit") or "").strip()
    st.session_state.borrador = None
    if texto:
        enviar_pregunta(texto)


def descartar_borrador():
    st.session_state.borrador = None


def panel_borrador():
    """Muestra lo que entendió el micrófono para que el alumno lo corrija antes de enviarlo."""
    if st.session_state.borrador is not None:
        st.text_input("✏️ Esto entendí. Corrígelo si hace falta y envía:", key="borrador_edit")
        c1, c2 = st.columns(2)
        c1.button("📨 Enviar pregunta", use_container_width=True, on_click=enviar_borrador, type="primary")
        c2.button("🗑️ Descartar", use_container_width=True, on_click=descartar_borrador)


def alternar_proyector():
    st.session_state.proyector = not st.session_state.proyector


def reiniciar():
    pausar_clase()
    st.session_state.slide = 1
    st.session_state.selector_slide = 1
    st.session_state.slide_anterior = 0
    st.session_state.historial = saludo_inicial()
    st.session_state.ultima_proy = None
    st.session_state.borrador = None
    st.session_state.responder_pendiente = False


def cambiar_curso():
    """Al elegir otro curso se reinicia todo (lámina 1, chat limpio) con el saludo del curso nuevo."""
    nuevo = st.session_state.selector_curso
    reiniciar()
    st.session_state.curso_id = nuevo
    st.session_state.historial = saludo_inicial(TITULOS[nuevo])


@st.cache_data(show_spinner=False)
def lamina_html(ruta: str, mtime: float, clase: str = "lamina-proy") -> str:
    """Imagen centrada y ajustada a la pantalla (HTML propio: no depende del DOM interno de Streamlit)."""
    from PIL import Image
    buf = io.BytesIO()
    with Image.open(ruta) as im:               # JPEG de calidad alta: ~4 veces más liviano que el PNG
        im = im.convert("RGB")
        im.thumbnail((1920, 1920))
        im.save(buf, "JPEG", quality=85, optimize=True)
    mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<div class="{clase}"><img src="data:{mime};base64,{b64}"></div>'


CSS_PROYECTOR = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {display: none !important;}
.block-container {padding: 0.4rem 1.2rem 0.4rem 1.2rem !important; max-width: 100% !important;}
.lamina-proy {display: flex; justify-content: center; align-items: center; width: 100%;}
.lamina-proy img {max-width: 100%; max-height: calc(100vh - 150px); width: auto; height: auto; object-fit: contain;}
</style>
"""


CSS_AULA = """
<style>
[data-testid="stDeployButton"], [data-testid="stToolbar"], #MainMenu, footer {display: none !important;}
.block-container {padding: 2.4rem 1.5rem 0.5rem 1.5rem !important; max-width: 100% !important;}
h1 {font-size: 1.7rem !important; padding: 0 0 0.3rem 0 !important;}
.lamina-aula {display: flex; justify-content: center; align-items: center; width: 100%; background: #f4f5f7;
              border: 1px solid #e3e5e8; border-radius: 12px; padding: 6px;}
.lamina-aula img {max-width: 100%; max-height: calc(100vh - 330px); width: auto; height: auto; object-fit: contain; border-radius: 8px;}
.stButton > button {border-radius: 10px; font-weight: 500;}
[data-testid="stSidebar"] .stButton > button {justify-content: flex-start; text-align: left;}
@media (max-width: 768px) {
  .block-container {padding: 1rem 0.6rem 0.5rem 0.6rem !important;}
  h1 {font-size: 1.2rem !important;}
  .lamina-aula img {max-height: 42vh;}
}
</style>
"""


def barra_atajos():
    """Botón de pantalla completa + atajos de teclado (flechas / RePág / AvPág de un control remoto, y tecla F)."""
    components.html("""
    <button id="fs" style="padding:6px 14px;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer;font-size:14px">
        ⛶ Pantalla completa
    </button>
    <span style="font-size:12px;color:#666;margin-left:10px">Teclas: ← → cambian de lámina · F pantalla completa</span>
    <script>
      const P = window.parent;
      function pantallaCompleta() {
        const d = P.document;
        if (d.fullscreenElement) { d.exitFullscreen(); } else { d.documentElement.requestFullscreen(); }
      }
      document.getElementById('fs').addEventListener('click', pantallaCompleta);
      if (!P.__aulaTeclas) {
        P.__aulaTeclas = true;
        P.document.addEventListener('keydown', function (e) {
          const t = ((e.target && e.target.tagName) || '').toLowerCase();
          if (t === 'input' || t === 'textarea') return;
          const clic = function (txt) {
            for (const b of P.document.querySelectorAll('button')) {
              if (b.innerText.includes(txt) && !b.disabled) { b.click(); return; }
            }
          };
          if (e.key === 'ArrowRight' || e.key === 'PageDown') clic('Siguiente');
          else if (e.key === 'ArrowLeft' || e.key === 'PageUp') clic('Anterior');
          else if (e.key === 'f' || e.key === 'F') pantallaCompleta();
        });
      }
    </script>
    """, height=44)


init_estado()

# =======================================================
# INTERFAZ
# =======================================================
proy = st.session_state.proyector
if not proy:
    st.markdown(CSS_AULA, unsafe_allow_html=True)

if not proy:
    st.title(f"🎓 Aula Virtual SST — {CURSO['titulo']}")

if not proy and not st.session_state.get("groq_ok"):
    err = verificar_groq()
    if err:
        st.error(err + "\n\n(La clase dictada funciona sin la IA, pero el asistente de dudas no.)")
    else:
        st.session_state.groq_ok = True

# ---- Barra lateral ----
with st.sidebar:
    st.button("📽️ Modo proyector", use_container_width=True, on_click=alternar_proyector, type="primary")
    st.divider()
    if len(CURSOS) > 1:
        st.selectbox("📚 Curso", options=list(CURSOS.keys()), format_func=lambda k: TITULOS[k],
                     key="selector_curso", on_change=cambiar_curso)
        st.divider()
    st.header("🎓 Acciones rápidas")
    st.button("🧒 Explícamelo más simple", use_container_width=True, on_click=enviar_pregunta,
              args=("Explícame esta lámina de forma más simple, como si fuera mi primer día en la empresa.",))
    st.button("🚚 Dame un ejemplo de mi trabajo", use_container_width=True, on_click=enviar_pregunta,
              args=("Dame un ejemplo práctico de esta lámina en mi trabajo diario.",))
    extra = CURSO.get("boton_extra")
    if isinstance(extra, dict) and extra.get("pregunta"):
        st.button(extra.get("etiqueta", "❓ Pregunta frecuente"), use_container_width=True,
                  on_click=enviar_pregunta, args=(extra["pregunta"],))
    else:
        st.button("📌 Lo más importante", use_container_width=True, on_click=enviar_pregunta,
                  args=("Resume en pocas líneas lo más importante de esta lámina.",))
    st.button("📝 Hazme una pregunta (quiz)", use_container_width=True, on_click=enviar_pregunta,
              args=("Hazme UNA sola pregunta de práctica sobre esta lámina y espera mi respuesta. "
                    "No des la respuesta todavía. Cuando yo responda, dime si estoy correcto y explica brevemente.",))
    st.divider()
    st.selectbox("Ir a la lámina", options=list(GUIONES.keys()),
                 format_func=lambda k: f"{k}. {GUIONES[k]['titulo']}",
                 key="selector_slide",
                 on_change=lambda: ir_a_slide(st.session_state.selector_slide))
    st.button("🔄 Reiniciar todo", use_container_width=True, on_click=reiniciar)
    listos = sum(_ruta_cache(g["guion"]).exists() for g in GUIONES.values())
    st.caption(f"🔊 Audio listo: {listos} de {TOTAL} láminas")
    if listos < TOTAL and st.button("⚡ Preparar audio de todas las láminas", use_container_width=True):
        prog = st.progress(0.0)
        for k, g in GUIONES.items():
            prog.progress(min(k / TOTAL, 1.0), text=f"Generando audio {k} de {TOTAL}...")
            audio_de_texto(g["guion"])
        prog.empty()
        st.rerun()
    st.caption(f"Asistente: Groq (`{MODELO}`) · Voz→texto: `{STT_MOTOR}` · Texto→voz: `{TTS_MOTOR}`")

chat_box = None
resp_box = None
audio_grabado = None
texto_chat = None

if proy:
    # ---------- MODO PROYECTOR: solo la lámina, grande, y controles mínimos ----------
    st.markdown(CSS_PROYECTOR, unsafe_allow_html=True)
    col_clase = st.container()
    col_chat = None
    with col_clase:
        idx = st.session_state.slide - 1
        if idx < len(LAMINAS):
            ruta = LAMINAS[idx]
            st.markdown(lamina_html(str(ruta), ruta.stat().st_mtime), unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        q1.button("⬅️ Anterior", use_container_width=True, on_click=cambiar_slide, args=(-1,),
                  disabled=st.session_state.slide <= 1)
        if st.session_state.modo_clase:
            q2.button("⏸️ Pausar clase", use_container_width=True, on_click=pausar_clase, type="primary")
        else:
            q2.button("▶️ Iniciar / Reanudar", use_container_width=True, on_click=iniciar_clase, type="primary")
        q3.button("Siguiente ➡️", use_container_width=True, on_click=cambiar_slide, args=(1,),
                  disabled=st.session_state.slide >= TOTAL)
        q4.button("✖ Salir del proyector", use_container_width=True, on_click=alternar_proyector)
        st.progress(st.session_state.slide / TOTAL)
        barra_atajos()
        player_box = st.container()

        # Pregunta por voz también en modo proyector
        m1, m2 = st.columns([0.28, 0.72])
        with m1:
            audio_grabado = mic_recorder(start_prompt="🎤 Preguntar con voz", stop_prompt="🟢 Enviar pregunta",
                                         just_once=True, format="wav", key="mic_proy")
        resp_box = m2.container()
        if audio_grabado and audio_grabado.get("bytes"):
            with st.spinner("🎙️ Transcribiendo..."):
                texto = transcribir_audio(audio_grabado["bytes"])
            if texto:
                guardar_borrador(texto)
            else:
                st.warning("No pude entender el audio, intenta de nuevo.")
                if st.session_state.get("error_audio"):
                    st.caption(f"Detalle técnico: {st.session_state.pop('error_audio')}")
        with resp_box:
            panel_borrador()
        if st.session_state.ultima_proy and not st.session_state.responder_pendiente:
            with resp_box:
                st.markdown(f"**🎤 Pregunta:** {st.session_state.ultima_proy[0]}")
                st.write(st.session_state.ultima_proy[1])
else:
    # ---------- MODO NORMAL (instructor) ----------
    col_clase, col_chat = st.columns([0.58, 0.42])

    with col_clase:
        b1, b2, b3, b4 = st.columns(4)
        if st.session_state.modo_clase:
            b1.button("✋ Tengo una duda", use_container_width=True, on_click=pausar_clase, type="primary")
        else:
            b1.button("▶️ Iniciar / Reanudar clase", use_container_width=True, on_click=iniciar_clase, type="primary")
        b2.button("🔊 Escuchar lámina", use_container_width=True, on_click=escuchar_lamina)
        b3.button("⬅️ Anterior", use_container_width=True, on_click=cambiar_slide, args=(-1,),
                  disabled=st.session_state.slide <= 1)
        b4.button("Siguiente ➡️", use_container_width=True, on_click=cambiar_slide, args=(1,),
                  disabled=st.session_state.slide >= TOTAL)

        estado = ":green[● Clase en curso]" if st.session_state.modo_clase else ":orange[● Clase en pausa]"
        st.markdown(f"**Lámina {st.session_state.slide} de {TOTAL}** — {GUIONES[st.session_state.slide]['titulo']} &nbsp;·&nbsp; {estado}")
        st.progress(st.session_state.slide / TOTAL)

        idx = st.session_state.slide - 1
        if idx < len(LAMINAS):
            st.markdown(lamina_html(str(LAMINAS[idx]), LAMINAS[idx].stat().st_mtime, "lamina-aula"), unsafe_allow_html=True)
        player_box = st.container()
        with st.expander("📜 Ver el texto que dicta el profesor en esta lámina"):
            st.write(GUIONES[st.session_state.slide]["guion"])

    with col_chat:
        st.subheader("💬 Dudas al Profesor")
        chat_box = st.container(height=440)
        audio_grabado = mic_recorder(start_prompt="🔴 Grabar voz", stop_prompt="🟢 Enviar audio",
                                     just_once=True, format="wav", key="mic_aula")
        st.caption("📱 En el celular, si el micrófono no responde, usa el micrófono del teclado en el cuadro de texto.")
        texto_chat = st.chat_input("Escribe tu duda aquí...")

        if audio_grabado and audio_grabado.get("bytes"):
            with st.spinner("🎙️ Transcribiendo..."):
                texto = transcribir_audio(audio_grabado["bytes"])
            if texto:
                guardar_borrador(texto)
            else:
                st.warning("No pude entender el audio, intenta de nuevo.")
                if st.session_state.get("error_audio"):
                    st.caption(f"Detalle técnico: {st.session_state.pop('error_audio')}")
        elif texto_chat:
            enviar_pregunta(texto_chat)
        panel_borrador()

# ---- Lógica: narrar la lámina al cambiar (modo clase) ----
if st.session_state.slide != st.session_state.slide_anterior:
    st.session_state.slide_anterior = st.session_state.slide
    narrar = st.session_state.modo_clase or st.session_state.pop("forzar_narracion", False)
    st.session_state.forzar_narracion = False
    if narrar and not st.session_state.responder_pendiente:
        with col_clase:
            with st.spinner("🎙️ Preparando la voz del profesor..."):
                datos, mime = audio_de_texto(GUIONES[st.session_state.slide]["guion"])
        if datos:
            st.session_state.audio_actual = {"bytes": datos, "mime": mime,
                                             "auto_avanzar": st.session_state.modo_clase}
            st.session_state.autoplay_pendiente = True
            if st.session_state.modo_clase and st.session_state.slide < TOTAL:
                precargar_en_segundo_plano(GUIONES[st.session_state.slide + 1]["guion"])
        else:
            st.session_state.audio_actual = None
            st.session_state.modo_clase = False
            with col_clase:
                st.error("No pude generar la voz. Si usas 'edge', necesitas internet; prueba TTS_MOTOR = 'pyttsx3'.")
    else:
        st.session_state.audio_actual = None   # cambió de lámina con la clase en pausa: silencio

# ---- Chat: historial + respuesta con streaming ----
if chat_box is not None:
    with chat_box:
        for msg in st.session_state.historial:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if st.session_state.responder_pendiente:
            st.session_state.responder_pendiente = False
            pregunta = st.session_state.historial[-1]["content"]
            contexto = st.session_state.historial[:-1]
            with st.chat_message("assistant"):
                try:
                    respuesta = st.write_stream(stream_profesor(pregunta, st.session_state.slide, contexto))
                except Exception as e:
                    respuesta = f"El asistente no está disponible. Revisa tu API Key de Groq o conexión. ({e})"
                    st.write(respuesta)
            st.session_state.historial.append({"role": "assistant", "content": respuesta})
            datos, mime = audio_de_texto(respuesta, usar_cache=False)
            if datos:
                st.session_state.audio_actual = {"bytes": datos, "mime": mime, "auto_avanzar": False}
                st.session_state.autoplay_pendiente = True

elif resp_box is not None and st.session_state.responder_pendiente:
    st.session_state.responder_pendiente = False
    pregunta = st.session_state.historial[-1]["content"]
    contexto = st.session_state.historial[:-1]
    with resp_box:
        st.markdown(f"**🎤 Pregunta:** {pregunta}")
        try:
            respuesta = st.write_stream(stream_profesor(pregunta, st.session_state.slide, contexto))
        except Exception as e:
            respuesta = f"El asistente no está disponible. Revisa tu API Key de Groq o conexión. ({e})"
            st.write(respuesta)
    st.session_state.historial.append({"role": "assistant", "content": respuesta})
    st.session_state.ultima_proy = (pregunta, respuesta)
    datos, mime = audio_de_texto(respuesta, usar_cache=False)
    if datos:
        st.session_state.audio_actual = {"bytes": datos, "mime": mime, "auto_avanzar": False}
        st.session_state.autoplay_pendiente = True

# ---- Reproductor (autoplay solo cuando el audio es nuevo) ----
with player_box:
    a = st.session_state.audio_actual
    if a:
        reproductor(a["bytes"], a["mime"],
                    autoplay=st.session_state.pop("autoplay_pendiente", False),
                    auto_avanzar=a["auto_avanzar"] and st.session_state.modo_clase)

# ---- Precalentar la imagen de la siguiente lámina (ya con la pantalla dibujada) ----
if st.session_state.slide < len(LAMINAS):
    _sig = LAMINAS[st.session_state.slide]
    lamina_html(str(_sig), _sig.stat().st_mtime, "lamina-proy" if proy else "lamina-aula")  
