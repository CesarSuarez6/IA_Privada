import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import requests
import threading
import time
import logging
import json
import os
from sentence_transformers import SentenceTransformer
import faiss  # Importa la biblioteca FAISS
import numpy as np  # FAISS usa NumPy
import csv
from PyPDF2 import PdfReader

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- VARIABLES DE CONFIGURACIÓN ---
DIRECTORIO_BASE_CONOCIMIENTO = r"C:\Users\PC9\Desktop\Emcrustador"  # Directorio con los archivos de texto de la empresa
INDICE_FAISS = r"C:\Users\PC9\Desktop\Emcrustador\mi_indice.faiss"  # Archivo para guardar el índice FAISS
OLLAMA_URL = "http://10.0.8.204:11434/api/generate"  # URL del servidor Ollama
MODELO_PREDETERMINADO = "test"  # Modelo predeterminado para usar en Ollama
ARCHIVO_MODELOS = "models.json"  # Archivo JSON con la lista de modelos disponibles

class ChatGUI:
    def __init__(self, master, ollama_url="http://10.0.8.204:11434/api/generate",
                 model_name="test", models=None):
        self.master = master
        master.title("Chat con Ollama")
        self.master.geometry("800x700")

        self.ollama_url = ollama_url
        self.models = models or ["test", "mistral", "codellama"]
        self.model_name = tk.StringVar(value=model_name)

        # Parámetros del modelo, con valores por defecto
        self.temperature = tk.DoubleVar(value=0.7)
        self.top_p = tk.DoubleVar(value=0.9)
        self.max_tokens = tk.IntVar(value=200)

        self.chat_log = scrolledtext.ScrolledText(master, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 12))
        self.chat_log.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.input_frame = tk.Frame(master)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.prompt_entry = tk.Entry(self.input_frame, font=("Arial", 12))
        self.prompt_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.send_button = ttk.Button(self.input_frame, text="Enviar", command=self.send_prompt)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        self.prompt_entry.bind("<Return>", lambda event: self.send_prompt())  # Enviar con Enter

        # Selección de Modelo
        self.model_label = tk.Label(master, text="Modelo:", font=("Arial", 10))
        self.model_label.pack()

        self.model_dropdown = ttk.Combobox(master, textvariable=self.model_name, values=self.models, state="readonly")
        self.model_dropdown.pack()

        # Parámetros de Generación
        self.params_frame = tk.LabelFrame(master, text="Parámetros de Generación", font=("Arial", 10))
        self.params_frame.pack(fill=tk.X, padx=10, pady=5)

        # Temperatura (Aleatoriedad)
        self.temp_label = tk.Label(self.params_frame, text="Aleatoriedad (Temperatura):", font=("Arial", 10))
        self.temp_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.temp_scale = tk.Scale(self.params_frame, variable=self.temperature, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, font=("Arial", 8))
        self.temp_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        # Top P (Predecibilidad)
        self.top_p_label = tk.Label(self.params_frame, text="Predecibilidad (Top P):", font=("Arial", 10))
        self.top_p_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.top_p_scale = tk.Scale(self.params_frame, variable=self.top_p, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, font=("Arial", 8))
        self.top_p_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

       # Max Tokens (Longitud)
        self.max_tokens_label = tk.Label(self.params_frame, text="Longitud Máxima (Tokens):", font=("Arial", 10))
        self.max_tokens_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_tokens_scale = tk.Scale(self.params_frame, variable=self.max_tokens, from_=50, to=500, resolution=10, orient=tk.HORIZONTAL, font=("Arial", 8))
        self.max_tokens_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)

        # Configuración de weights del grid para que se expanda bien
        self.params_frame.columnconfigure(1, weight=1)

        # Indicador "Escribiendo..."
        self.typing_indicator = tk.Label(master, text="", font=("Arial", 10))
        self.typing_indicator.pack(pady=5)
        self.dots = 0
        self.typing_animation_id = None #Para controlar la animación

        # Manejar el cierre de la ventana
        #master.protocol("WM_DELETE_WINDOW", self.on_closing)

        #Caché Manual
        self.cache = {}
        self.cache_lock = threading.Lock()

        #Cargar el modelo de embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Cargar el índice FAISS y los documentos
        self.faiss_index = None
        self.documents = []  # Guarda los documentos originales
        self.load_faiss_index()

    def send_prompt(self):
        prompt = self.prompt_entry.get()
        self.prompt_entry.delete(0, tk.END)

        self.add_to_chat_log(f"Tú: {prompt}\n", "user")

        # Mostrar el indicador "Escribiendo..."
        self.typing_indicator_text = "Ollama está pensando"
        self.typing_indicator.config(text=self.typing_indicator_text)
        self.start_typing_animation()

        # Enviar el prompt al servidor Ollama en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self.get_ollama_response, args=(prompt,)).start()

    def start_typing_animation(self):
        self.dots = 0
        self.animate_typing_indicator()

    def animate_typing_indicator(self):
        self.dots = (self.dots + 1) % 4  # 0, 1, 2, 3, 0, 1, ...
        dots_str = "." * self.dots
        self.typing_indicator.config(text=self.typing_indicator_text + dots_str)
        self.typing_animation_id = self.master.after(500, self.animate_typing_indicator)  # Llama a la función cada 500ms

    def stop_typing_animation(self):
        if self.typing_animation_id:
            self.master.after_cancel(self.typing_animation_id) #Cancela la animación
            self.typing_animation_id = None
        self.typing_indicator.config(text="") #Limpia el label

    def get_ollama_response(self, prompt):
        start_time = time.time()
        model = self.model_name.get()
        temperature = self.temperature.get()
        top_p = self.top_p.get()
        max_tokens = self.max_tokens.get()

        try:
            # 1. Generar el embedding de la consulta
            query_embedding = self.embedding_model.encode(prompt) #Devuelve un numpy array

            # 2. Buscar documentos similares en FAISS
            if self.faiss_index:  # Solo busca si el índice FAISS se cargó correctamente
                D, I = self.faiss_index.search(np.expand_dims(query_embedding, axis=0).astype('float32'), k=3) #FAISS espera un array de tipo float32
                context = "\n".join([self.documents[i] for i in I[0]]) #Recupera los documentos del array I
            else:
                context = "No se pudo cargar la base de conocimiento de la empresa."  # Mensaje si no hay base de conocimiento

            # 3. Construir el prompt con el contexto
            final_prompt = f"""Utiliza el siguiente contexto para responder a la pregunta.
            Si la respuesta no se encuentra en el contexto, responde que no tienes la información.

            Contexto:
            {context}

            Pregunta:
            {prompt}
            """

            data = {
                "prompt": final_prompt,
                "model": model,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens
                }
            }
            response = self.make_request(data)
            response_json = response.json()
            ollama_response = response_json.get("response", "Sin respuesta")

            # Actualizar la interfaz de usuario en el hilo principal
            self.master.after(0, self.add_to_chat_log, f"Ollama ({model}): {ollama_response}\n", "ollama")


        except requests.exceptions.RequestException as e:
            logging.error(f"Error de conexión con Ollama: {e}")
            self.master.after(0, self.show_error_message, f"Error de conexión con Ollama: {e}")
            self.master.after(0, self.add_to_chat_log, f"Error: Error de conexión con Ollama: {e}\n", "error")

        except Exception as e:
            logging.exception(f"Error inesperado: {e}")
            self.master.after(0, self.show_error_message, f"Error inesperado: {e}")
            self.master.after(0, self.add_to_chat_log, f"Error: Error inesperado: {e}\n", "error")

        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time_str = f" ({elapsed_time:.2f} segundos)"

            self.master.after(0, self.stop_typing_animation)
            self.master.after(0, self.typing_indicator.config, {"text": f"Ollama tardó: {elapsed_time_str}"})


    def make_request(self, data):
        """Realiza la petición al servidor Ollama con reintento automático."""
        max_retries = 3
        retry_delay = 1 #segundos
        for attempt in range(max_retries):
            try:
                response = requests.post(self.ollama_url, json=data)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logging.warning(f"Intento {attempt + 1} fallido: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay) #Espera antes de reintentar
                else:
                    raise #Si falla despues de varios intentos, levanta la excepcion para que se maneje en `get_ollama_response`
            except Exception as e:
                logging.exception(f"Fallo al hacer la peticion por: {e}")
                raise

    def add_to_chat_log(self, text, tag=None):
        self.chat_log.config(state=tk.NORMAL)  # Habilitar la edición temporalmente
        self.chat_log.insert(tk.END, text, tag)  # Insertar el texto con la etiqueta opcional
        self.chat_log.config(state=tk.DISABLED)  # Deshabilitar la edición de nuevo
        self.chat_log.see(tk.END)  # Scroll al final

        # Configurar colores para diferentes etiquetas
        if tag == "user":
            self.chat_log.tag_config("user", foreground="blue")
        elif tag == "ollama":
            self.chat_log.tag_config("ollama", foreground="green")
        elif tag == "error":
            self.chat_log.tag_config("error", foreground="red")

    def show_error_message(self, message):
         messagebox.showerror("Error", message)

    def on_closing(self):
        """Ejecuta el comando 'ollama stop' para detener el modelo."""
        try:
            model = self.model_name.get()  # Obtiene el nombre del modelo
            subprocess.run(['ollama', 'stop', model], check=True, capture_output=True, text=True)
            logging.info(f"Modelo '{model}' detenido correctamente.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error al detener el modelo: {e.stderr}")
            self.show_error_message(f"Error al detener el modelo: {e.stderr}")
        except FileNotFoundError:
            logging.error("El comando 'ollama' no se encuentra. Asegúrate de que Ollama está instalado y en el PATH.")
            self.show_error_message("El comando 'ollama' no se encuentra. Asegúrate de que Ollama está instalado y en el PATH.")
        except Exception as e:
            logging.exception(f"Error inesperado al detener el modelo: {e}")
            self.show_error_message(f"Error inesperado al detener el modelo: {e}")
        finally:
            self.master.destroy()  # Cierra la ventana principal

    def load_faiss_index(self):
        """Carga el índice FAISS desde el disco."""
        try:
            if os.path.exists(INDICE_FAISS):
                self.faiss_index = faiss.read_index(INDICE_FAISS)
                logging.info("Índice FAISS cargado desde el disco.")
                #Cargar documentos relacionados al texto
                with open("documentos.json", "r") as f:
                   self.documents = json.load(f)
            else:
                logging.warning("Índice FAISS no encontrado. Por favor, crea la base de conocimiento primero.")
                self.show_error_message("Índice FAISS no encontrado. Por favor, crea la base de conocimiento primero.")
                self.faiss_index = None
        except Exception as e:
            logging.error(f"Error al cargar el índice FAISS: {e}")
            self.show_error_message(f"Error al cargar el índice FAISS: {e}")
            self.faiss_index = None

def cargar_base_de_conocimiento(directorio=DIRECTORIO_BASE_CONOCIMIENTO, indice_faiss=INDICE_FAISS):
    """
    Lee archivos de texto de un directorio, genera embeddings y crea un índice FAISS.

    Args:
        directorio (str): El directorio que contiene los archivos de texto.
        indice_faiss (str): El archivo donde se guardará el índice FAISS.
    """
    try:
        # 1. Leer los documentos del directorio
        documents = []
        document_ids = []
        for filename in os.listdir(directorio):
            if filename.lower().endswith(".txt"):
                filepath = os.path.join(directorio, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    document = f.read()
                    documents.append(document)
                    document_ids.append(filename)
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(directorio, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        document = f.read()
                except Exception as e:
                    #Intenta leer PDF
                    try:
                        reader = PdfReader(filepath)
                        number_of_pages = len(reader.pages)
                        document = ""
                        for page_number in range(number_of_pages):
                            page = reader.pages[page_number]
                            document += page.extract_text()
                    except Exception as e:
                        logging.warning(f"No se pudo leer {filename}: {e}")
                        continue

                documents.append(document)
                document_ids.append(filename)
            if filename.lower().endswith(".csv"):
                filepath = os.path.join(directorio, filename)

                try:
                    with open(filepath, 'r', encoding="utf-8") as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader)  # Leer la fila de encabezado

                        # Construir un texto unificado combinando encabezado y filas
                        text = ''
                        for row in reader:
                            # Crea un diccionario combinando los encabezados y los valores de la fila
                            row_dict = {header[i].strip(): row[i].strip() for i in range(len(header))}
                            # Formatea el diccionario como una cadena de texto
                            row_text = ' '.join([f'{key}: {value}' for key, value in row_dict.items()])
                            text += f"{{{row_text}}} \n " #Añade llaves para que sea mas facil de identificar como un objeto

                        #return text
                except Exception as e:
                    logging.warning(f"No se pudo leer el archivo CSV {filepath}: {e}")
                    #return None
    
                documents.append(text)
                document_ids.append(filename)

        if not documents:
            raise FileNotFoundError(f"No se encontraron archivos .txt en el directorio: {directorio}")

        # 2. Generar los embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(documents)
        dimension = embeddings.shape[1] #Obtenemos la dimension

        # 3. Crear el índice FAISS
        index = faiss.IndexFlatL2(dimension)  # Usar L2 distance (euclidean)
        faiss.normalize_L2(embeddings) #Normalizamos
        index.add(embeddings.astype('float32')) #FAISS espera tipo float32

        # 4. Guardar el índice FAISS
        faiss.write_index(index, indice_faiss)
        logging.info(f"Índice FAISS guardado en {indice_faiss}")

        #Guardar los documentos originales (necesario para la busqueda)
        with open("documentos.json", "w") as f:
          json.dump(documents, f)

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.exception(f"Error inesperado: {e}")

def main():
    root = tk.Tk()

    # Intenta leer la lista de modelos del archivo
    try:
        with open(ARCHIVO_MODELOS, "r") as f:
            modelos = json.load(f)
    except FileNotFoundError:
            logging.warning("Archivo models.json no encontrado, usando lista por defecto")
            modelos =  ["test", "mistral", "codellama"] #lista por defecto
    except json.JSONDecodeError:
            logging.error("Error al leer models.json, revisa que sea un json valido")
            modelos =  ["test", "mistral", "codellama"] #lista por defecto

    #Primero carga la base de conocimiento
    cargar_base_de_conocimiento()
    # Crea la instancia de ChatGUI y carga la base de conocimiento
    gui = ChatGUI(root, models=modelos)
    root.mainloop()

if __name__ == "__main__":
    main()